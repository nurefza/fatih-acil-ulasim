"""
20_fatih_acil_ulasim.py
Fatih Acil Ulaşım Rehberi — yeni, bağımsız Streamlit sürümü.

Bu dosya:
- Başlangıç noktasını Fatih mahalle listesinden seçtirir.
- Hastane / polis karakolu / resmî toplanma alanı türünü ve belirli hedefi seçtirir.
- AFAD/e-Devlet bağlantısına ihtiyaç duymadan yerleşik resmî toplanma alanı listesini kullanır.
- Rota maliyetinde yol uzunluğu, fiziksel kapanma riski ve trafik baskısını birlikte kullanır.

Gerekli mevcut proje dosyaları:
- veri/fatih_yol_agi.graphml
- veri/trafik_tipoloji.gpkg (veya veri/tahmin.gpkg)
- fatih_mahalle_hasar.gpkg (veya veri/fatih_mahalle_hasar.gpkg)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import subprocess
import sys
import time

import folium
import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium


# ============================================================
# UYGULAMA AYARLARI
# ============================================================

PLACE_NAME = "Fatih, Istanbul, Turkey"

KAPANMA_RISK_AGIRLIK = 4.0
TRAFIK_BASKISI_AGIRLIK = 3.0

TESIS_TURLERI = {
    "hastane": "Hastane",
    "karakol": "Polis Karakolu",
    "toplanma_alani": "Resmî Toplanma Alanı",
}

TIPOLOJI_ETIKETI = {
    "1. Kritik Darboğaz": "Acil müdahale önceliği",
    "2. Fiziksel Risk": "Yüksek kapanma riski",
    "3. Trafik Öncelikli": "Yüksek trafik baskısı",
    "4. Düşük Öncelik": "Düşük öncelik",
}

TIPOLOJI_RENGI = {
    "1. Kritik Darboğaz": "#C62828",
    "2. Fiziksel Risk": "#EF6C00",
    "3. Trafik Öncelikli": "#1565C0",
    "4. Düşük Öncelik": "#BDBDBD",
}

RISK_RENKLERI = {
    "Düşük": "#2E7D32",
    "Orta": "#F9A825",
    "Yüksek": "#C62828",
}

TRAFIK_RENKLERI = {
    "Düşük": "#2E7D32",
    "Orta": "#F9A825",
    "Yüksek": "#7B1FA2",
}

st.set_page_config(
    page_title="Fatih Acil Ulaşım Rehberi",
    page_icon="🧭",
    layout="wide",
)


# ============================================================
# GENEL YARDIMCI FONKSİYONLAR
# ============================================================

def guvenli_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def metin(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def normalize(value: Any) -> str:
    text = metin(value).lower()
    table = str.maketrans(
        {
            "ı": "i", "İ": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g",
            "ü": "u", "Ü": "u", "ö": "o", "Ö": "o", "ç": "c", "Ç": "c",
        }
    )
    return text.translate(table)


def first_matching_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {normalize(col): col for col in columns}
    for candidate in candidates:
        found = normalized.get(normalize(candidate))
        if found is not None:
            return found
    return None


def risk_duzeyi(value: float) -> str:
    if value < 0.33:
        return "Düşük"
    if value < 0.66:
        return "Orta"
    return "Yüksek"


def edge_value(
    u: Any,
    v: Any,
    lookup: dict[tuple[str, str], float],
    default: float = 0.0,
) -> float:
    return lookup.get(
        (str(u), str(v)),
        lookup.get((str(v), str(u)), default),
    )


def edge_data_for_route(graph: nx.MultiDiGraph, u: Any, v: Any) -> dict:
    edges = graph.get_edge_data(u, v)
    if not edges:
        return {}

    return min(
        edges.values(),
        key=lambda item: guvenli_float(item.get("length", 1.0), 1.0),
    )


def edge_name(graph: nx.MultiDiGraph, u: Any, v: Any) -> str:
    data = edge_data_for_route(graph, u, v)
    name = data.get("name", "Bağlantı yolu")

    if isinstance(name, list):
        name = name[0] if name else "Bağlantı yolu"

    name = metin(name)
    return name if name else "Bağlantı yolu"


def geometry_as_point(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    result = gdf.copy()

    if result.crs is None:
        result = result.set_crs(4326, allow_override=True)
    else:
        result = result.to_crs(4326)

    result = result[result.geometry.notna()].copy()
    result["geometry"] = result.geometry.apply(
        lambda geom: geom
        if geom is not None and geom.geom_type == "Point"
        else geom.representative_point()
        if geom is not None
        else None
    )
    return result[result.geometry.notna()].copy()


# ============================================================
# RESMÎ VE İSİMLİ KRİTİK TESİS VERİSİ
# ============================================================

# Hastane listesi Fatih Kaymakamlığı'nın ilçe sağlık kurumları listesi temel
# alınarak hazırlanmıştır. Koordinat, yalnızca seçilen hedef için OSM/Nominatim
# üzerinden çözülür; böylece uygulama açılırken gereksiz istek atılmaz.
HASTANE_LISTESI = [
    {
        "ad": "İstanbul Üniversitesi İstanbul Tıp Fakültesi Hastanesi (Çapa)",
        "sorgular": [
            "İstanbul Tıp Fakültesi Hastanesi, Fatih, İstanbul",
            "Çapa Hastanesi, Fatih, İstanbul",
        ],
    },
    {
        "ad": "İstanbul Üniversitesi-Cerrahpaşa Cerrahpaşa Tıp Fakültesi Hastanesi",
        "sorgular": [
            "Cerrahpaşa Tıp Fakültesi Hastanesi, Fatih, İstanbul",
            "Cerrahpaşa Hastanesi, Fatih, İstanbul",
        ],
    },
    {
        "ad": "Bezmiâlem Vakıf Üniversitesi Hastanesi",
        "sorgular": [
            "Bezmiâlem Vakıf Üniversitesi Hastanesi, Fatih, İstanbul",
            "Bezmiâlem Vakıf Üniversitesi Tıp Fakültesi Hastanesi, Fatih, İstanbul",
        ],
    },
    {
        "ad": "Haseki Eğitim ve Araştırma Hastanesi",
        "sorgular": [
            "Haseki Eğitim ve Araştırma Hastanesi, Fatih, İstanbul",
            "Haseki Hastanesi, Fatih, İstanbul",
        ],
    },
    {
        "ad": "İstanbul Eğitim ve Araştırma Hastanesi",
        "sorgular": [
            "İstanbul Eğitim ve Araştırma Hastanesi, Fatih, İstanbul",
            "Samatya Eğitim ve Araştırma Hastanesi, Fatih, İstanbul",
        ],
    },
    {
        "ad": "Özel Çapa Hastanesi",
        "sorgular": [
            "Özel Çapa Hastanesi, Fatih, İstanbul",
            "Çapa Hastanesi, Fatih, İstanbul",
        ],
    },
    {
        "ad": "Medilife Fatih Hastanesi",
        "sorgular": [
            "Medilife Fatih Hastanesi, Fatih, İstanbul",
            "Medilife, Fatih, İstanbul",
        ],
    },
    {
        "ad": "Medical Park Fatih Hastanesi",
        "sorgular": [
            "Medical Park Fatih Hastanesi, Fatih, İstanbul",
            "Medical Park, Fatih, İstanbul",
        ],
    },
    {
        "ad": "Özel Fatih Hastanesi",
        "sorgular": [
            "Özel Fatih Hastanesi, Fatih, İstanbul",
            "Fatih Hastanesi, Fatih, İstanbul",
        ],
    },
    {
        "ad": "Medipol Üniversitesi Hastanesi",
        "sorgular": [
            "Medipol Üniversitesi Hastanesi, Fatih, İstanbul",
            "Medipol Hastanesi, Fatih, İstanbul",
        ],
    },
]

# Fatih Belediyesi'nin yayımladığı 2019 tarihli resmî toplanma alanı listesi.
# AFAD/e-Devlet uç noktası geçici olarak yanıt vermediğinde uygulamanın açılmasını
# engellememek için bu liste uygulama içine alınmıştır.
RESMI_TOPLANMA_ALANLARI = [{'ad': 'Binbirdirek Parkı', 'mahalle': 'Binbirdirek Mh.', 'adres': 'Dr. Şevki Bey Sk.'}, {'ad': 'İMÇ Parkı', 'mahalle': 'Hacı Kadın Mh.', 'adres': 'Hacı Kadın Cad.'}, {'ad': 'Numune Parkı', 'mahalle': 'Kemal Paşa Mh.', 'adres': 'Ömer Yılmaz Sk.'}, {'ad': 'Kalburcu Mehmet Parkı', 'mahalle': 'Mevlanakapı Mh.', 'adres': 'Kalburcu Mehmet Çeşmesi Sk.'}, {'ad': 'Çemberlitaş Meydanı', 'mahalle': 'Molla Fenari Mh.', 'adres': 'Vezirhanı Cad.'}, {'ad': 'Yenicami Meydan Parkı', 'mahalle': 'Rüstem Paşa Mh.', 'adres': 'Bankacılar Sk.'}, {'ad': 'Şenol Güneş Parkı', 'mahalle': 'Akşemsettin Mh.', 'adres': 'Akdeniz Cad.'}, {'ad': 'Akşemsettin Parkı', 'mahalle': 'Akşemsettin Mh.', 'adres': 'Kocasinan Cad.'}, {'ad': 'Koyunbaba Parkı', 'mahalle': 'Akşemsettin Mh.', 'adres': 'Mütercim Asım Cad.'}, {'ad': 'Yavuz Selim Çocuk Parkı', 'mahalle': 'Atikali Mh.', 'adres': 'Yavuz Selim Cad.'}, {'ad': 'Ayvansaray Mahkemealtı Parkı', 'mahalle': 'Ayvansaray Mh.', 'adres': 'Mahkemaltı Cad.'}, {'ad': 'Çarşamba Çukurbostan Parkı', 'mahalle': 'Balat Mh.', 'adres': 'Sultan Selim Cad.'}, {'ad': 'Fatih Anıt Parkı', 'mahalle': 'Zeyrek Mh.', 'adres': 'İtfaiye Cad.'}, {'ad': 'Fındıkzade Çukurbostan Şehir Parkı', 'mahalle': 'Seyyid Ömer Mh.', 'adres': 'Sırrıpaşa Sk.'}, {'ad': 'Mehmet Akif Ersoy Parkı', 'mahalle': 'Binbirdirek Mh.', 'adres': 'Divanyolu Cad.'}, {'ad': 'Sultanahmet Meydanı', 'mahalle': 'Binbirdirek Mh.', 'adres': 'At Meydanı Cad.'}, {'ad': 'Edirnekapı Meydanı', 'mahalle': 'Derviş Ali Mh.', 'adres': 'Fevzipaşa Cad.'}, {'ad': 'Namık Sevik Stadı', 'mahalle': 'Sümbül Efendi Mh.', 'adres': 'Hisaraltı 1. Cad.'}, {'ad': 'Kocamustafapaşa Meydanı', 'mahalle': 'Silivrikapı Mh.', 'adres': 'Kuvayi Milliye Cad.'}, {'ad': 'Gülhane Parkı', 'mahalle': 'Cankurtaran Mh.', 'adres': 'Alemdar Cad.'}, {'ad': 'Karagümrük Stadı', 'mahalle': 'Derviş Ali Mh.', 'adres': 'Fevzipaşa Cad.'}, {'ad': 'Aksaray Metro İstasyonu', 'mahalle': 'İskenderpaşa Mh.', 'adres': 'Vatan Cad.'}, {'ad': 'Beyazıt Meydanı', 'mahalle': 'Beyazıt Mh.', 'adres': 'Yeniçeriler Cad.'}, {'ad': 'Fatih Camii Avlusu', 'mahalle': 'Ali Kuşçu Mh.', 'adres': 'Fevzipaşa Cad.'}, {'ad': 'Mimar Sinan Stadı', 'mahalle': 'Karagümrük Mh.', 'adres': 'Keçeci Meydanı Sk.'}, {'ad': 'Kemikliburun Parkı', 'mahalle': 'Mevlanakapı Mh.', 'adres': 'Kemikliburun Sk.'}, {'ad': 'Tekfur Sarayı Parkı', 'mahalle': 'Ayvansaray Mh.', 'adres': 'Hoca Çakır Cad.'}, {'ad': 'Avcıbey Parkı', 'mahalle': 'Ayvansaray Mh.', 'adres': 'Şişhane Cad.'}, {'ad': 'Molla Aşkı Parkı', 'mahalle': 'Ayvansaray Mh.', 'adres': 'Çınçınlı Çeşme Sk.'}, {'ad': 'Melek Hatun Parkı', 'mahalle': 'Mevlanakapı Mh.', 'adres': 'Hasırcı Melek Sk.'}, {'ad': 'İbrahim Çavuş Parkı', 'mahalle': 'Mevlanakapı Mh.', 'adres': 'Keresteci Veli Sk.'}, {'ad': 'Kırımlı Parkı', 'mahalle': 'Mevlanakapı Mh.', 'adres': 'Simkeş Cami Sk.'}, {'ad': 'Silivrikapı Set Üstü Parkı', 'mahalle': 'Mevlanakapı Mh.', 'adres': 'Hancı Değirmen Sk.'}, {'ad': 'Mahmut Celalettin Ökten Meydanı', 'mahalle': 'Ayvansaray Mh.', 'adres': 'Püsküllü Cad.'}, {'ad': 'Arkeoloji Parkı', 'mahalle': 'İskenderpaşa Mh.', 'adres': 'Kavalalı Sk.'}, {'ad': 'Saraçhane Parkı', 'mahalle': 'Kalenderhane Mh.', 'adres': '15 Temmuz Şehitleri Cad.'}, {'ad': 'Çarşamba Meydanı', 'mahalle': 'Balat Mh.', 'adres': 'Lokmacı Dede Sk.'}, {'ad': 'Kurtağa Parkı', 'mahalle': 'Derviş Ali Mh.', 'adres': 'Kurtağa Çeşmesi Sk.'}, {'ad': 'Dervişali Parkı', 'mahalle': 'Derviş Ali Mh.', 'adres': 'Sena Sk.'}, {'ad': 'Kariye Meydan Parkı', 'mahalle': 'Derviş Ali Mh.', 'adres': 'Feyzullah Efendi Sk.'}, {'ad': 'Kariye Şehir Parkı', 'mahalle': 'Derviş Ali Mh.', 'adres': 'Karıye Bostanı Sk.'}, {'ad': 'Karagümrük Çocuk Parkı', 'mahalle': 'Hırka-i Şerif Mh.', 'adres': 'Karagümrük Meydanı Sk.- Karabulut Sk.'}, {'ad': 'Haşim İşcan Parkı', 'mahalle': 'Karagümrük Mh.', 'adres': 'Sofalı Çeşme Cad.'}, {'ad': 'Özgüven Parkı', 'mahalle': 'Karagümrük Mh.', 'adres': 'Dumlupınar Sk.'}, {'ad': 'Küçük Mustafapaşa Parkı', 'mahalle': 'Yavuz Sultan Selim Mh.', 'adres': 'Kalaycı Sk.'}, {'ad': 'Muhtar Osman Güven Parkı', 'mahalle': 'Yavuz Sultan Selim Mh.', 'adres': 'Dinibütün Sk.'}, {'ad': 'Şair Nabi Parkı', 'mahalle': 'Yavuz Sultan Selim Mh.', 'adres': 'Kopça Sk.'}, {'ad': 'Kadıçeşme Parkı', 'mahalle': 'Zeyrek Mh.', 'adres': 'Haliçgören Sk.'}, {'ad': 'Hekimoğlu Ali Paşa Parkı', 'mahalle': 'Cerrahpaşa Mh.', 'adres': 'Hekimoğlu Alipaşa Cad.'}, {'ad': 'Keyci Hatun Parkı', 'mahalle': 'Cerrahpaşa Mh.', 'adres': 'Haseki Kadın Sk.'}, {'ad': 'Dr. Metin Alatlı Parkı', 'mahalle': 'Haseki Sultan Mh.', 'adres': 'Suphipaşa Sk.'}, {'ad': 'Veledi Karabaş Parkı', 'mahalle': 'Mevlanakapı Mh.', 'adres': 'Aynalı Bakkal Sk.'}, {'ad': 'Kamil Başaran Parkı', 'mahalle': 'Seyyid Ömer Mh.', 'adres': 'Miralay Hasan Kazımbey Sk.'}, {'ad': 'Seyyid Ömer Şelaleli Park', 'mahalle': 'Seyyid Ömer Mh.', 'adres': 'Emrullah Efendi Sk.'}, {'ad': 'Taşköprülü Parkı', 'mahalle': 'Seyyid Ömer Mh.', 'adres': 'Hüseyin Kazım Sk.'}, {'ad': 'Seyyid Ömer Meydanı', 'mahalle': 'Seyyid Ömer Mh.', 'adres': 'Vezir Cad.'}, {'ad': 'Şehit Ast. Furkan Işık Parkı', 'mahalle': 'Seyyid Ömer Mh.', 'adres': 'Cevdetpaşa Cad.'}, {'ad': 'Uzunyusuf Parkı', 'mahalle': 'Seyyid Ömer Mh.', 'adres': 'Lalezar Cami Sk.'}, {'ad': 'Mustafa Nafi Parkı', 'mahalle': 'Seyyid Ömer Mh.', 'adres': 'Mustafa Nafi Sk.'}, {'ad': 'Silivrikapı Semt Parkı', 'mahalle': 'Silivrikapı Mh.', 'adres': 'Koçdibek Sk.'}, {'ad': 'Kocamustafapaşa Semt Parkı', 'mahalle': 'Silivrikapı Mh.', 'adres': 'Şehit Turan Topal Sk.'}, {'ad': 'Ramazan Efendi Cami Arkası Parkı', 'mahalle': 'Silivrikapı Mh.', 'adres': 'Bezirgan Odaları SK.'}, {'ad': 'Silivrikapı Alay İmamı Parkı', 'mahalle': 'Silivrikapı Mh.', 'adres': 'Alay İmamı Sk.'}, {'ad': 'Çukurbostan Parkı', 'mahalle': 'Şehremini Mh.', 'adres': 'Ziya Gökalp Sk.'}, {'ad': 'Büyük Saray Meydan Parkı', 'mahalle': 'Silivrikapı Mh.', 'adres': 'Büyük Saray Meydanı Cad.'}, {'ad': 'Vezir Parkı', 'mahalle': 'Silivrikapı Mh.', 'adres': 'Vezir Cad.'}, {'ad': 'Sefa Bostan Parkı', 'mahalle': 'Topkapı Mh.', 'adres': 'Sefa Bostanı Sk.'}, {'ad': 'Yedikule Yeşil Alanları', 'mahalle': 'Yedikule Mh.', 'adres': 'Yedikule Cad.'}, {'ad': 'Yedikule Parkı', 'mahalle': 'Yedikule Mh.', 'adres': 'Hacı Piri Sk.'}, {'ad': 'Yedikule Sur Parkı', 'mahalle': 'Yedikule Mh.', 'adres': 'Yedikule Meydanı Sk.'}, {'ad': 'Muratpaşa Parkı', 'mahalle': 'Molla Gürani Mh.', 'adres': 'Vatan Cad.'}, {'ad': 'Molla Şeref Parkı', 'mahalle': 'Molla Gürani Mh.', 'adres': 'Tomrukçu Sk.'}, {'ad': 'Vatan Cad. Meydanı', 'mahalle': 'Molla Gürani Mh.', 'adres': 'Oğuzhan Cad.'}, {'ad': 'Selçuk Sultan Parkı', 'mahalle': 'Molla Gürani Mh.', 'adres': 'Selçuk Sultan Cami Sk.'}, {'ad': 'Şehit Mehmet Çetinkaya Parkı', 'mahalle': 'Molla Gürani Mh.', 'adres': 'Şehit Pilot Mahmut Nedim Sk.'}, {'ad': 'Engelliler Parkı', 'mahalle': 'Molla Gürani Mh.', 'adres': 'Dr. Ahmetpaşa Sk.'}, {'ad': 'Nakilbent Parkı', 'mahalle': 'Küçük Ayasofya Mh.', 'adres': 'Nakilbent Sk.'}]


def osm_noktasi_bul(sorgular: list[str]) -> tuple[float, float] | None:
    """Nominatim ile sırayla verilen arama metinlerini dener."""
    for query in sorgular:
        try:
            lat, lon = ox.geocode(query)
            return float(lat), float(lon)
        except Exception:
            time.sleep(0.25)
    return None


def resmi_hastaneleri_olustur() -> pd.DataFrame:
    """Tüm hastaneleri menüye ekler; koordinatı seçildiğinde çözülür."""
    rows = []
    for hospital in HASTANE_LISTESI:
        rows.append(
            {
                "ad": hospital["ad"],
                "tip": "hastane",
                "enlem": np.nan,
                "boylam": np.nan,
                "mahalle": "",
                "adres": "",
                "sorgular": "||".join(hospital["sorgular"]),
                "kaynak": "Fatih Kaymakamlığı + OpenStreetMap",
            }
        )
    return pd.DataFrame(rows)


def osm_polis_karakollarini_olustur() -> pd.DataFrame:
    """OpenStreetMap'ten isim ve koordinatı bulunan polis birimlerini alır."""
    try:
        gdf = ox.features_from_place(PLACE_NAME, tags={"amenity": "police"})
    except Exception as error:
        raise RuntimeError(
            f"Polis karakolu verisi OpenStreetMap'ten alınamadı: {error}"
        ) from error

    if gdf.empty:
        return pd.DataFrame(
            columns=["ad", "tip", "enlem", "boylam", "mahalle", "adres", "sorgular", "kaynak"]
        )

    gdf = geometry_as_point(gdf)
    name_col = first_matching_column(
        list(gdf.columns),
        ["name:tr", "name", "official_name", "alt_name"],
    )

    if name_col is None:
        return pd.DataFrame(
            columns=["ad", "tip", "enlem", "boylam", "mahalle", "adres", "sorgular", "kaynak"]
        )

    rows = []
    for _, row in gdf.iterrows():
        name = metin(row.get(name_col))
        if not name:
            continue

        rows.append(
            {
                "ad": name,
                "tip": "karakol",
                "enlem": round(float(row.geometry.y), 6),
                "boylam": round(float(row.geometry.x), 6),
                "mahalle": "",
                "adres": "",
                "sorgular": "",
                "kaynak": "OpenStreetMap",
            }
        )

    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["ad", "enlem", "boylam"])
        .reset_index(drop=True)
    )


def resmi_toplanma_alanlarini_olustur() -> pd.DataFrame:
    """AFAD bağlantısı olmadan yerleşik belediye listesini menüye ekler."""
    result = pd.DataFrame(RESMI_TOPLANMA_ALANLARI).copy()
    result["tip"] = "toplanma_alani"
    result["enlem"] = np.nan
    result["boylam"] = np.nan
    result["sorgular"] = result.apply(
        lambda row: "||".join(
            [
                f"{row['ad']}, {row['adres']}, {row['mahalle']}, Fatih, İstanbul",
                f"{row['ad']}, Fatih, İstanbul",
            ]
        ),
        axis=1,
    )
    result["kaynak"] = "Fatih Belediyesi (2019 resmî liste)"
    return result[
        ["ad", "tip", "enlem", "boylam", "mahalle", "adres", "sorgular", "kaynak"]
    ]


def tesis_katalogunu_olustur() -> pd.DataFrame:
    """
    Yeni veri şeması:
      - Hastane: ilçe sağlık kurumları listesi
      - Polis karakolu: OpenStreetMap isimli kayıtları
      - Toplanma alanı: Fatih Belediyesi resmî yayımlı liste
    """
    hospitals = resmi_hastaneleri_olustur()
    police = osm_polis_karakollarini_olustur()
    assembly = resmi_toplanma_alanlarini_olustur()

    facilities = pd.concat([hospitals, police, assembly], ignore_index=True)
    facilities["ad"] = facilities["ad"].map(metin)
    facilities = facilities[facilities["ad"].ne("")]
    facilities = facilities.drop_duplicates(subset=["tip", "ad"])
    return facilities.sort_values(["tip", "ad"]).reset_index(drop=True)


def tesisleri_yukle() -> pd.DataFrame:
    """v3 katalog, önceki AFAD/e-Devlet önbelleklerini bilinçli olarak kullanmaz."""
    cache_path = Path("veri/kritik_tesisler_katalogu_v3.csv")
    cache_path.parent.mkdir(exist_ok=True)

    if cache_path.exists():
        facilities = pd.read_csv(cache_path)
    else:
        facilities = tesis_katalogunu_olustur()
        facilities.to_csv(cache_path, index=False, encoding="utf-8-sig")

    needed = {"ad", "tip", "enlem", "boylam", "mahalle", "adres", "sorgular"}
    if not needed.issubset(facilities.columns):
        raise ValueError(
            "veri/kritik_tesisler_katalogu_v3.csv dosyasında gerekli sütunlar bulunamadı."
        )

    facilities = facilities.copy()
    facilities["ad"] = facilities["ad"].map(metin)
    facilities["tip"] = facilities["tip"].map(normalize)
    facilities["enlem"] = pd.to_numeric(facilities["enlem"], errors="coerce")
    facilities["boylam"] = pd.to_numeric(facilities["boylam"], errors="coerce")
    facilities["mahalle"] = facilities["mahalle"].map(metin)
    facilities["adres"] = facilities["adres"].map(metin)
    facilities["sorgular"] = facilities["sorgular"].map(metin)

    facilities = facilities[
        facilities["tip"].isin(TESIS_TURLERI)
        & facilities["ad"].ne("")
    ].drop_duplicates(subset=["tip", "ad"])

    if facilities.empty:
        raise RuntimeError("Gerçek isimli tesis kaydı oluşturulamadı.")

    facilities = facilities.sort_values(["tip", "ad"]).reset_index(drop=True)
    facilities["tesis_id"] = facilities.index.astype(str)
    facilities["hedef_anahtar"] = (
        facilities["tip"] + "|" + facilities["ad"] + "|" + facilities["mahalle"] + "|" + facilities["adres"]
    )
    return facilities


def hedef_koordinatini_coz(
    facility: pd.Series,
    graph: nx.MultiDiGraph,
) -> pd.Series:
    """
    Seçilen hedefin koordinatı yoksa yalnızca o seçimi OSM/Nominatim ile çözer.
    Başarılı çözüm veri/hedef_koordinat_onbellegi_v3.csv dosyasına kaydedilir.
    """
    target = facility.copy()
    lat = guvenli_float(target.get("enlem"), np.nan)
    lon = guvenli_float(target.get("boylam"), np.nan)

    if np.isfinite(lat) and np.isfinite(lon):
        target["node"] = ox.distance.nearest_nodes(graph, X=lon, Y=lat)
        return target

    cache_path = Path("veri/hedef_koordinat_onbellegi_v3.csv")
    key = metin(target.get("hedef_anahtar"))

    if cache_path.exists():
        try:
            cache = pd.read_csv(cache_path)
            found = cache[cache["hedef_anahtar"] == key]
            if not found.empty:
                lat = guvenli_float(found.iloc[0]["enlem"], np.nan)
                lon = guvenli_float(found.iloc[0]["boylam"], np.nan)
                if np.isfinite(lat) and np.isfinite(lon):
                    target["enlem"] = lat
                    target["boylam"] = lon
                    target["node"] = ox.distance.nearest_nodes(graph, X=lon, Y=lat)
                    return target
        except Exception:
            pass

    queries = [item.strip() for item in metin(target.get("sorgular")).split("||") if item.strip()]
    if not queries:
        queries = [f"{target['ad']}, Fatih, İstanbul"]

    with st.spinner("Seçilen hedefin konumu bulunuyor..."):
        point = osm_noktasi_bul(queries)

    if point is None:
        raise RuntimeError(
            "Seçilen hedefin konumu bulunamadı. Lütfen listeden başka bir alan seçin."
        )

    lat, lon = point
    target["enlem"] = lat
    target["boylam"] = lon
    target["node"] = ox.distance.nearest_nodes(graph, X=lon, Y=lat)

    new_row = pd.DataFrame(
        [{"hedef_anahtar": key, "enlem": lat, "boylam": lon}]
    )
    if cache_path.exists():
        try:
            existing = pd.read_csv(cache_path)
            existing = existing[existing["hedef_anahtar"] != key]
            new_row = pd.concat([existing, new_row], ignore_index=True)
        except Exception:
            pass
    new_row.to_csv(cache_path, index=False, encoding="utf-8-sig")

    return target


# ============================================================
# MAHALLE VE ANALİZ VERİLERİ
# ============================================================

def mahalleleri_yukle(graph: nx.MultiDiGraph) -> pd.DataFrame:
    candidates = [
        Path("fatih_mahalle_hasar.gpkg"),
        Path("veri/fatih_mahalle_hasar.gpkg"),
        Path("mahalle.gpkg"),
        Path("veri/mahalle.gpkg"),
    ]

    path = next((candidate for candidate in candidates if candidate.exists()), None)

    if path is None:
        raise FileNotFoundError(
            "Mahalle dosyası bulunamadı. fatih_mahalle_hasar.gpkg dosyasını proje "
            "ana klasöründe veya veri klasöründe kontrol edin."
        )

    mahalle = gpd.read_file(path)
    name_col = first_matching_column(
        list(mahalle.columns),
        ["mahalle", "mahalle_adi", "name", "adi", "ad"],
    )

    if name_col is None:
        raise ValueError("Mahalle dosyasında mahalle adını içeren sütun bulunamadı.")

    if mahalle.crs is None:
        mahalle = mahalle.set_crs(4326, allow_override=True)
    else:
        mahalle = mahalle.to_crs(4326)

    mahalle = mahalle[mahalle.geometry.notna()].copy()
    mahalle["temsil"] = mahalle.geometry.representative_point()

    output = pd.DataFrame(
        {
            "mahalle": mahalle[name_col].map(metin),
            "enlem": mahalle["temsil"].y,
            "boylam": mahalle["temsil"].x,
        }
    ).dropna(subset=["enlem", "boylam"])

    output = output[output["mahalle"].ne("")]
    output = output.drop_duplicates(subset=["mahalle"]).sort_values("mahalle")
    output["mahalle_id"] = output.index.astype(str)

    output["node"] = ox.distance.nearest_nodes(
        graph,
        X=output["boylam"].values,
        Y=output["enlem"].values,
    )

    return output.reset_index(drop=True)


@st.cache_resource(show_spinner="Veriler yükleniyor...")
def verileri_yukle() -> tuple[
    nx.MultiDiGraph,
    gpd.GeoDataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[tuple[str, str], float],
    dict[tuple[str, str], float],
    pd.DataFrame | None,
]:
    graph_path = Path("veri/fatih_yol_agi.graphml")
    typology_path = Path("veri/trafik_tipoloji.gpkg")
    prediction_path = Path("veri/tahmin.gpkg")

    if not graph_path.exists():
        raise FileNotFoundError("veri/fatih_yol_agi.graphml bulunamadı.")

    if typology_path.exists():
        segments = gpd.read_file(typology_path)
    elif prediction_path.exists():
        segments = gpd.read_file(prediction_path)
    else:
        raise FileNotFoundError(
            "veri/trafik_tipoloji.gpkg veya veri/tahmin.gpkg bulunamadı."
        )

    needed = {"u", "v", "kapanma_olasiligi"}
    missing = needed - set(segments.columns)
    if missing:
        raise ValueError(
            f"Segment verisinde eksik sütun var: {', '.join(sorted(missing))}"
        )

    graph = ox.load_graphml(graph_path)

    if segments.crs is None:
        segments = segments.set_crs(4326, allow_override=True)
    else:
        segments = segments.to_crs(4326)

    segments = segments.copy()
    segments["geometry"] = segments.geometry.simplify(
        0.00002,
        preserve_topology=False,
    )

    segments["kapanma_riski"] = pd.to_numeric(
        segments["kapanma_olasiligi"],
        errors="coerce",
    ).fillna(0.0)

    if "trafik_tikaniklik" in segments.columns:
        segments["trafik_baskisi"] = pd.to_numeric(
            segments["trafik_tikaniklik"],
            errors="coerce",
        ).fillna(0.0)
    else:
        segments["trafik_baskisi"] = 0.0

    traffic_median = segments["trafik_baskisi"].median()

    if "segment_tipolojisi" not in segments.columns:
        segments["segment_tipolojisi"] = np.select(
            [
                (segments["kapanma_riski"] >= 0.50)
                & (segments["trafik_baskisi"] >= traffic_median),
                (segments["kapanma_riski"] >= 0.50)
                & (segments["trafik_baskisi"] < traffic_median),
                (segments["kapanma_riski"] < 0.50)
                & (segments["trafik_baskisi"] >= traffic_median),
            ],
            [
                "1. Kritik Darboğaz",
                "2. Fiziksel Risk",
                "3. Trafik Öncelikli",
            ],
            default="4. Düşük Öncelik",
        )

    facilities = tesisleri_yukle()
    facilities["node"] = None
    koordinatli = facilities[facilities["enlem"].notna() & facilities["boylam"].notna()].copy()
    if not koordinatli.empty:
        facilities.loc[koordinatli.index, "node"] = ox.distance.nearest_nodes(
            graph,
            X=koordinatli["boylam"].values,
            Y=koordinatli["enlem"].values,
        )

    neighborhoods = mahalleleri_yukle(graph)

    closure_lookup: dict[tuple[str, str], float] = {}
    traffic_lookup: dict[tuple[str, str], float] = {}

    for row in segments[
        ["u", "v", "kapanma_riski", "trafik_baskisi"]
    ].itertuples(index=False):
        closure_lookup[(str(row.u), str(row.v))] = float(row.kapanma_riski)
        traffic_lookup[(str(row.u), str(row.v))] = float(row.trafik_baskisi)

    summary_path = Path("tablolar/mahalle_bazli_risk_ozeti.csv")
    neighborhood_summary = (
        pd.read_csv(summary_path) if summary_path.exists() else None
    )

    return (
        graph,
        segments,
        facilities,
        neighborhoods,
        closure_lookup,
        traffic_lookup,
        neighborhood_summary,
    )


# ============================================================
# HARİTA VE ROTA
# ============================================================

def harita_katmani_hazirla(
    segments: gpd.GeoDataFrame,
    map_mode: str,
) -> tuple[gpd.GeoDataFrame, str]:
    data = segments.copy()

    if map_mode == "Müdahale öncelikleri":
        data["etiket"] = data["segment_tipolojisi"].map(TIPOLOJI_ETIKETI)
        data["renk"] = data["segment_tipolojisi"].map(TIPOLOJI_RENGI).fillna("#BDBDBD")
        data["kalinlik"] = data["segment_tipolojisi"].map(
            {
                "1. Kritik Darboğaz": 3.2,
                "2. Fiziksel Risk": 2.4,
                "3. Trafik Öncelikli": 2.0,
                "4. Düşük Öncelik": 1.0,
            }
        ).fillna(1.0)
        return data, "Müdahale düzeyi"

    if map_mode == "Kapanma riski":
        data["etiket"] = data["kapanma_riski"].apply(risk_duzeyi)
        data["renk"] = data["etiket"].map(RISK_RENKLERI)
        data["kalinlik"] = data["etiket"].map(
            {"Düşük": 1.0, "Orta": 2.0, "Yüksek": 3.0}
        )
        return data, "Kapanma riski"

    data["etiket"] = data["trafik_baskisi"].apply(risk_duzeyi)
    data["renk"] = data["etiket"].map(TRAFIK_RENKLERI)
    data["kalinlik"] = data["etiket"].map(
        {"Düşük": 1.0, "Orta": 2.0, "Yüksek": 3.0}
    )
    return data, "Trafik yoğunluğu"


def rota_hesapla(
    graph: nx.MultiDiGraph,
    start_node: Any,
    target_node: Any,
    closure_lookup: dict[tuple[str, str], float],
    traffic_lookup: dict[tuple[str, str], float],
) -> tuple[list[Any], list[dict[str, Any]], dict[str, float]]:
    for u, v, key, data in graph.edges(keys=True, data=True):
        length = guvenli_float(data.get("length", 1.0), 1.0)
        closure = edge_value(u, v, closure_lookup)
        traffic = edge_value(u, v, traffic_lookup)
        data["rota_maliyeti"] = length * (
            1
            + KAPANMA_RISK_AGIRLIK * closure
            + TRAFIK_BASKISI_AGIRLIK * traffic
        )

    route_nodes = nx.shortest_path(
        graph,
        source=start_node,
        target=target_node,
        weight="rota_maliyeti",
        method="dijkstra",
    )

    details: list[dict[str, Any]] = []
    lengths: list[float] = []
    closure_values: list[float] = []
    traffic_values: list[float] = []

    for u, v in zip(route_nodes[:-1], route_nodes[1:]):
        edge = edge_data_for_route(graph, u, v)
        length = guvenli_float(edge.get("length", 1.0), 1.0)
        closure = edge_value(u, v, closure_lookup)
        traffic = edge_value(u, v, traffic_lookup)

        lengths.append(length)
        closure_values.append(closure)
        traffic_values.append(traffic)

        details.append(
            {
                "Yol": edge_name(graph, u, v),
                "Uzunluk (m)": length,
                "Kapanma riski": risk_duzeyi(closure),
                "Trafik yoğunluğu": risk_duzeyi(traffic),
                "_kapanma": closure,
                "_trafik": traffic,
            }
        )

    weights = lengths if sum(lengths) > 0 else None
    metrics = {
        "uzunluk_km": sum(lengths) / 1000,
        "kapanma": float(np.average(closure_values, weights=weights)),
        "trafik": float(np.average(traffic_values, weights=weights)),
        "dikkat_kapanma": int(sum(x >= 0.66 for x in closure_values)),
        "dikkat_trafik": int(sum(x >= 0.66 for x in traffic_values)),
    }

    return route_nodes, details, metrics


def haritayi_olustur(
    graph: nx.MultiDiGraph,
    map_segments: gpd.GeoDataFrame,
    tooltip_title: str,
    facilities: pd.DataFrame,
    selected_type: str,
    selected_facility_id: str,
    start_row: pd.Series,
    resolved_target: pd.Series | None,
    route_nodes: list[Any] | None,
    closure_lookup: dict[tuple[str, str], float],
    traffic_lookup: dict[tuple[str, str], float],
) -> folium.Map:
    fmap = folium.Map(
        location=[41.012, 28.949],
        zoom_start=14,
        tiles="cartodbpositron",
        control_scale=True,
    )

    folium.GeoJson(
        map_segments[
            ["geometry", "etiket", "renk", "kalinlik"]
        ].to_json(),
        style_function=lambda feature: {
            "color": feature["properties"]["renk"],
            "weight": feature["properties"]["kalinlik"],
            "opacity": 0.80,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["etiket"],
            aliases=[f"{tooltip_title}:"],
            localize=True,
            sticky=False,
        ),
        name="Yol ağı",
    ).add_to(fmap)

    visible = facilities[
        (facilities["tip"] == selected_type)
        & facilities["enlem"].notna()
        & facilities["boylam"].notna()
    ].copy()

    for facility in visible.itertuples():
        selected = facility.tesis_id == selected_facility_id
        folium.CircleMarker(
            location=[facility.enlem, facility.boylam],
            radius=8 if selected else 4,
            color="#4A148C" if selected else "#1E5AA8",
            fill=True,
            fill_opacity=0.95 if selected else 0.70,
            tooltip=(
                f"Seçilen hedef: {facility.ad}"
                if selected
                else facility.ad
            ),
        ).add_to(fmap)

    if resolved_target is not None:
        target_lat = guvenli_float(resolved_target.get("enlem"), np.nan)
        target_lon = guvenli_float(resolved_target.get("boylam"), np.nan)
        if np.isfinite(target_lat) and np.isfinite(target_lon):
            folium.Marker(
                location=[target_lat, target_lon],
                tooltip=f"Seçilen hedef: {resolved_target['ad']}",
                icon=folium.Icon(color="red", icon="flag", prefix="fa"),
            ).add_to(fmap)

    folium.Marker(
        location=[start_row["enlem"], start_row["boylam"]],
        tooltip=f"Başlangıç mahallesi: {start_row['mahalle']}",
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
    ).add_to(fmap)

    if route_nodes:
        folium.PolyLine(
            [
                (graph.nodes[node]["y"], graph.nodes[node]["x"])
                for node in route_nodes
            ],
            color="#202020",
            weight=9,
            opacity=0.42,
        ).add_to(fmap)

        for u, v in zip(route_nodes[:-1], route_nodes[1:]):
            closure = edge_value(u, v, closure_lookup)
            traffic = edge_value(u, v, traffic_lookup)

            route_color = (
                "#C62828"
                if closure >= 0.66
                else "#F9A825"
                if traffic >= 0.66
                else "#1976D2"
            )

            folium.PolyLine(
                [
                    (graph.nodes[u]["y"], graph.nodes[u]["x"]),
                    (graph.nodes[v]["y"], graph.nodes[v]["x"]),
                ],
                color=route_color,
                weight=5,
                opacity=1.0,
            ).add_to(fmap)

    folium.LayerControl(collapsed=True).add_to(fmap)
    return fmap


# ============================================================
# UYGULAMA
# ============================================================

def main() -> None:
    try:
        (
            graph,
            segments,
            facilities,
            neighborhoods,
            closure_lookup,
            traffic_lookup,
            neighborhood_summary,
        ) = verileri_yukle()
    except Exception as error:
        st.error(f"Uygulama verileri yüklenemedi: {error}")
        st.stop()

    if "rota_anahtari" not in st.session_state:
        st.session_state.rota_anahtari = None

    with st.sidebar:
        st.header("Güzergâh Planla")

        st.caption("1. Başlangıç mahallesini seçin.")
        selected_neighborhood_id = st.selectbox(
            "Başlangıç mahallesi",
            neighborhoods["mahalle_id"].tolist(),
            format_func=lambda value: neighborhoods.loc[
                neighborhoods["mahalle_id"] == value, "mahalle"
            ].iloc[0],
        )
        start_row = neighborhoods[
            neighborhoods["mahalle_id"] == selected_neighborhood_id
        ].iloc[0]

        st.caption("2. Ulaşmak istediğiniz yeri seçin.")
        selected_type = st.selectbox(
            "Ulaşmak istediğiniz tesis türü",
            list(TESIS_TURLERI),
            format_func=lambda value: TESIS_TURLERI[value],
        )

        options = facilities[facilities["tip"] == selected_type].copy()
        options = options.sort_values("ad")

        if options.empty:
            st.warning(
                "Bu tesis türü için isimli kayıt bulunamadı. "
                "Veri/kritik_tesisler_adli.csv dosyasını silip uygulamayı yenileyin."
            )
            st.stop()

        selected_facility_id = st.selectbox(
            "Gitmek istediğiniz yer",
            options["tesis_id"].tolist(),
            format_func=lambda value: options.loc[
                options["tesis_id"] == value, "ad"
            ].iloc[0],
        )

        selected_facility = options[
            options["tesis_id"] == selected_facility_id
        ].iloc[0]

        route_key = f"{selected_neighborhood_id}|{selected_facility_id}"

        if st.button("Daha düşük riskli güzergâhı göster", type="primary"):
            st.session_state.rota_anahtari = route_key

        st.divider()
        st.header("Harita Görünümü")

        map_mode = st.radio(
            "Haritada neyi görmek istiyorsunuz?",
            ["Müdahale öncelikleri", "Kapanma riski", "Trafik yoğunluğu"],
        )

        descriptions = {
            "Müdahale öncelikleri": (
                "Hem kapanma riski hem de trafik baskısı yüksek yollar "
                "kırmızıyla gösterilir."
            ),
            "Kapanma riski": (
                "Yol çevresindeki fiziksel koşullara göre hesaplanan "
                "kapanma risk düzeyini gösterir."
            ),
            "Trafik yoğunluğu": (
                "2024 olağan koşullarındaki trafik baskısı düzeyini gösterir."
            ),
        }
        st.caption(descriptions[map_mode])

        st.divider()
        st.caption(
            "Bu araç gerçek zamanlı yol açıklığı veya kesin güvenlik garantisi vermez. "
            "Deprem öncesi hazırlık ve senaryo temelli planlama amacıyla geliştirilmiştir."
        )

    st.title("🧭 Fatih Acil Ulaşım Rehberi")
    st.caption(
        "İBB Mw 7,5 deprem senaryosu, yol çevresi özellikleri ve 2024 trafik verisi "
        "kullanılarak hazırlanmıştır. Toplanma alanı menüsü, Fatih Belediyesi’nin yayımladığı resmî listedeki alanları içerir."
    )

    total_segments = len(segments)
    critical_count = int(
        (segments["segment_tipolojisi"] == "1. Kritik Darboğaz").sum()
    )
    critical_share = (
        100 * critical_count / total_segments if total_segments else 0.0
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("İncelenen yol segmenti", f"{total_segments:,}")
    c2.metric(
        "Acil müdahale öncelikli koridor",
        f"{critical_count:,}",
        f"Yol ağının %{critical_share:.1f}'i",
    )
    c3.metric("Seçilen hedef", selected_facility["ad"])

    map_segments, tooltip_title = harita_katmani_hazirla(segments, map_mode)

    route_nodes: list[Any] | None = None
    route_details: list[dict[str, Any]] = []
    route_metrics: dict[str, float] | None = None
    resolved_target: pd.Series | None = None

    if st.session_state.rota_anahtari == route_key:
        try:
            resolved_target = hedef_koordinatini_coz(selected_facility, graph)
            route_nodes, route_details, route_metrics = rota_hesapla(
                graph=graph,
                start_node=start_row["node"],
                target_node=resolved_target["node"],
                closure_lookup=closure_lookup,
                traffic_lookup=traffic_lookup,
            )
        except nx.NetworkXNoPath:
            st.error("Seçilen mahalle ile hedef arasında yol ağı üzerinde rota bulunamadı.")
        except Exception as error:
            st.error(f"Güzergâh hesaplanamadı: {error}")

    fmap = haritayi_olustur(
        graph=graph,
        map_segments=map_segments,
        tooltip_title=tooltip_title,
        facilities=facilities,
        selected_type=selected_type,
        selected_facility_id=selected_facility_id,
        start_row=start_row,
        resolved_target=resolved_target,
        route_nodes=route_nodes,
        closure_lookup=closure_lookup,
        traffic_lookup=traffic_lookup,
    )

    left, right = st.columns([3.2, 1.35])

    with left:
        st.subheader("Harita")
        st.caption(
            f"Başlangıç: {start_row['mahalle']} mahallesi · "
            f"Hedef: {selected_facility['ad']}"
        )
        st_folium(fmap, width="stretch", height=650)

    with right:
        st.subheader("Güzergâh Özeti")

        if route_metrics is None:
            st.info(
                "Soldaki seçimleri tamamladıktan sonra “Daha düşük riskli güzergâhı göster” düğmesine basın."
            )
        else:
            st.success(
                f"{start_row['mahalle']} mahallesinden {selected_facility['ad']} için rota oluşturuldu."
            )

            st.metric("Tahmini yol uzunluğu", f"{route_metrics['uzunluk_km']:.1f} km")
            st.metric(
                "Güzergâhtaki kapanma riski",
                risk_duzeyi(route_metrics["kapanma"]),
            )
            st.metric(
                "Güzergâhtaki trafik yoğunluğu",
                risk_duzeyi(route_metrics["trafik"]),
            )

            if (
                route_metrics["dikkat_kapanma"] > 0
                or route_metrics["dikkat_trafik"] > 0
            ):
                st.warning(
                    "Rota, toplam riski düşürmek üzere seçilmiştir; "
                    "ancak ağda tamamen düşük riskli bir seçenek bulunmadığında "
                    "bazı dikkat gerektiren yol kesimleri içerebilir."
                )
            else:
                st.success(
                    "Bu güzergâhta yüksek kapanma riski veya yüksek trafik baskısı taşıyan yol kesimi görünmemektedir."
                )

            if route_details:
                route_df = pd.DataFrame(route_details)
                summary = (
                    route_df.groupby("Yol", as_index=False)
                    .agg(
                        {
                            "Uzunluk (m)": "sum",
                            "_kapanma": "mean",
                            "_trafik": "mean",
                        }
                    )
                    .sort_values("Uzunluk (m)", ascending=False)
                )

                summary["Kapanma riski"] = summary["_kapanma"].apply(risk_duzeyi)
                summary["Trafik yoğunluğu"] = summary["_trafik"].apply(risk_duzeyi)
                summary["Uzunluk (m)"] = summary["Uzunluk (m)"].round().astype(int)

                summary = summary[
                    ["Yol", "Uzunluk (m)", "Kapanma riski", "Trafik yoğunluğu"]
                ]

                with st.expander("Güzergâh ayrıntıları"):
                    st.dataframe(summary, hide_index=True, width="stretch")

    if neighborhood_summary is not None:
        required = {
            "mahalle",
            "kritik_darbogaz_segment",
            "kritik_darbogaz_orani_%",
        }

        if required.issubset(neighborhood_summary.columns):
            st.divider()
            st.subheader("Mahallelerde Müdahale Önceliği")
            st.caption(
                "Bu bölüm rota önerisinden bağımsızdır. İlçe genelinde acil müdahale "
                "öncelikli koridorların hangi mahallelerde yoğunlaştığını gösterir."
            )

            neighborhood_view = neighborhood_summary[
                [
                    "mahalle",
                    "kritik_darbogaz_segment",
                    "kritik_darbogaz_orani_%",
                ]
            ].copy()

            neighborhood_view = neighborhood_view.rename(
                columns={
                    "mahalle": "Mahalle",
                    "kritik_darbogaz_segment": "Acil öncelikli koridor sayısı",
                    "kritik_darbogaz_orani_%": "Acil öncelikli koridor oranı (%)",
                }
            )

            neighborhood_view["Acil öncelikli koridor oranı (%)"] = (
                neighborhood_view["Acil öncelikli koridor oranı (%)"].round(1)
            )

            neighborhood_view = neighborhood_view.sort_values(
                "Acil öncelikli koridor oranı (%)",
                ascending=False,
            )

            chart_data = neighborhood_view.head(10).set_index("Mahalle")
            st.bar_chart(
                chart_data["Acil öncelikli koridor oranı (%)"],
                height=320,
            )

            st.dataframe(
                neighborhood_view.head(15),
                hide_index=True,
                width="stretch",
            )


if __name__ == "__main__":
    main()
