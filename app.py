import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# ------------------------------
# Sayfa Ayarları
# ------------------------------
st.set_page_config(
    page_title="🚗 Türkiye Araç Fiyat Tahmin Sistemi",
    page_icon="🚗",
    layout="wide"
)

# ------------------------------
# Model ve Veriyi Yükle
# ------------------------------
@st.cache_resource
def load_data():
    model = joblib.load("arac_fiyat_modeli.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    df = pd.read_csv("temiz_veri.csv")
    return model, feature_columns, df

model, feature_columns, df = load_data()

# ------------------------------
# Başlık
# ------------------------------
st.title("🚗 Türkiye Araç Fiyat Tahmin Sistemi")

st.markdown("""
Bu uygulama **Random Forest** makine öğrenmesi algoritması kullanılarak geliştirilmiştir.

Aracınızın bilgilerini girerek **tahmini piyasa satış fiyatını** öğrenebilirsiniz.
""")

st.divider()

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.header("🚘 Araç Bilgileri")

marka = st.sidebar.selectbox(
    "Marka",
    sorted(df["Marka"].dropna().unique())
)

arac_tip_grubu = st.sidebar.selectbox(
    "Araç Tip Grubu",
    sorted(
        df[df["Marka"] == marka]["Arac_Tip_Grubu"].dropna().unique()
    )
)

arac_tip = st.sidebar.selectbox(
    "Araç Tipi",
    sorted(
        df[
            (df["Marka"] == marka)
            &
            (df["Arac_Tip_Grubu"] == arac_tip_grubu)
        ]["Arac_Tip"].dropna().unique()
    )
)

model_yil = st.sidebar.slider(
    "Model Yılı",
    min_value=1990,
    max_value=2020,
    value=2018
)

km = st.sidebar.number_input(
    "Kilometre",
    min_value=0,
    max_value=1000000,
    value=50000,
    step=5000
)

yakit = st.sidebar.selectbox(
    "Yakıt Türü",
    sorted(df["Yakit_Turu"].dropna().unique())
)

vites = st.sidebar.selectbox(
    "Vites",
    sorted(df["Vites"].dropna().unique())
)

kasa = st.sidebar.selectbox(
    "Kasa Tipi",
    sorted(df["Kasa_Tipi"].dropna().unique())
)

kimden = st.sidebar.selectbox(
    "Kimden",
    sorted(df["Kimden"].dropna().unique())
)

durum = st.sidebar.selectbox(
    "Durum",
    sorted(df["Durum"].dropna().unique())
)

st.divider()

# ------------------------------
# Tahmin
# ------------------------------
if st.button("🚀 Fiyat Tahmini Yap", use_container_width=True):

    arac_yasi = datetime.now().year - model_yil
    km_yil = km / max(arac_yasi, 1)

    user_data = {
        "Marka": marka,
        "Arac_Tip_Grubu": arac_tip_grubu,
        "Arac_Tip": arac_tip,
        "Model_Yil": model_yil,
        "Yakit_Turu": yakit,
        "Vites": vites,
        "Kasa_Tipi": kasa,
        "Kimden": kimden,
        "Durum": durum,
        "Km": km,
        "Arac_Yasi": arac_yasi,
        "Km_Yil": km_yil
    }

    user_df = pd.DataFrame([user_data])

    user_df = pd.get_dummies(user_df)

    for col in feature_columns:
        if col not in user_df.columns:
            user_df[col] = 0

    user_df = user_df[feature_columns]

    tahmin = model.predict(user_df)[0]

    alt = tahmin * 0.90
    ust = tahmin * 1.10

    st.success("✅ Tahmin Başarıyla Tamamlandı")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Alt Sınır",
        f"{alt:,.0f} TL"
    )

    c2.metric(
        "Tahmini Fiyat",
        f"{tahmin:,.0f} TL"
    )

    c3.metric(
        "Üst Sınır",
        f"{ust:,.0f} TL"
    )

    st.progress(90)

    st.info(
        "Tahmini doğruluk seviyesi yaklaşık %90'dır. "
        "Gerçek satış fiyatı hasar kaydı, donanım paketi, boya durumu ve pazarlık payına göre değişebilir."
    )

    st.subheader("📋 Girilen Araç Bilgileri")

    st.table(pd.DataFrame({
        "Özellik": [
            "Marka",
            "Araç Tip Grubu",
            "Araç Tipi",
            "Model Yılı",
            "Kilometre",
            "Yakıt",
            "Vites",
            "Kasa Tipi",
            "Kimden",
            "Durum"
        ],
        "Değer": [
            marka,
            arac_tip_grubu,
            arac_tip,
            model_yil,
            f"{km:,} km",
            yakit,
            vites,
            kasa,
            kimden,
            durum
        ]
    }))

    st.subheader("🚗 Veri Setindeki Benzer Araçlar")

    benzer = df[
        (df["Marka"] == marka)
        &
        (df["Arac_Tip_Grubu"] == arac_tip_grubu)
    ][
        ["Model_Yil", "Km", "Fiyat"]
    ].sort_values(
        by="Model_Yil",
        ascending=False
    ).head(10)

    st.dataframe(
        benzer,
        use_container_width=True
    )

st.divider()

with st.expander("ℹ️ Proje Hakkında"):

    st.write("""
Bu uygulama Türkiye ikinci el araç piyasasına ait geniş bir veri kümesi kullanılarak geliştirilmiştir.

Makine öğrenmesi modeli olarak **Random Forest Regressor** kullanılmıştır.

Model aşağıdaki özellikleri dikkate alarak tahmin üretmektedir:

- Marka
- Araç Tip Grubu
- Araç Tipi
- Model Yılı
- Yakıt Türü
- Vites
- Kasa Tipi
- Kilometre
- Kimden
- Durum
- Araç Yaşı
- Yıllık Kilometre

Bu uygulamanın amacı araç sahiplerine satış öncesinde yaklaşık piyasa değeri hakkında fikir vermektir.
""")

st.caption("© 2026 | Türkiye Araç Fiyat Tahmin Sistemi | Random Forest")
