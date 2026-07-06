import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 1) VERİYİ OKU
# ============================================================
df = pd.read_csv("turkey_car_market.csv")

print("İlk veri boyutu:", df.shape)

# ============================================================
# 2) TEMİZLEME
# ============================================================
df["Model Yıl"] = df["Model Yıl"].astype(int)

df["Fiyat"] = (
    df["Fiyat"]
    .astype(str)
    .str.replace("TL", "", regex=False)
    .str.replace(".", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)
df["Fiyat"] = pd.to_numeric(df["Fiyat"], errors="coerce")

df["Km"] = (
    df["Km"]
    .astype(str)
    .str.replace("km", "", regex=False)
    .str.replace(".", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)
df["Km"] = pd.to_numeric(df["Km"], errors="coerce")

df = df.drop_duplicates()

df = df[(df["Fiyat"] > 10000) & (df["Fiyat"] < 5000000)]
df = df[(df["Km"] >= 0) & (df["Km"] < 1000000)]

print("Temizleme sonrası veri boyutu:", df.shape)

# ============================================================
# 3) KOLON İSİMLERİNİ DÜZENLE
# ============================================================
df = df.rename(columns={
    "İlan Tarihi": "Ilan_Tarihi",
    "Marka": "Marka",
    "Arac Tip Grubu": "Arac_Tip_Grubu",
    "Arac Tip": "Arac_Tip",
    "Model Yıl": "Model_Yil",
    "Yakıt Turu": "Yakit_Turu",
    "Vites": "Vites",
    "CCM": "CCM",
    "Beygir Gucu": "Beygir_Gucu",
    "Renk": "Renk",
    "Kasa Tipi": "Kasa_Tipi",
    "Kimden": "Kimden",
    "Durum": "Durum",
    "Km": "Km",
    "Fiyat": "Fiyat"
})

# ============================================================
# 4) YENİ ÖZELLİKLER (FEATURE ENGINEERING)
# ============================================================
df["Arac_Yasi"] = 2020 - df["Model_Yil"]
df["Km_Yil"] = df["Km"] / df["Arac_Yasi"].replace(0, 1)

# ============================================================
# 5) NADİR KATEGORİLERİ GRUPLA
# ------------------------------------------------------------
# Bu adım dosya boyutunu küçültmenin en etkili yolu.
# Marka ve Arac_Tip kolonlarında çok az sayıda ilanı olan
# değerleri "Diger" altında topluyoruz. Böylece one-hot
# encoding sonrası sütun sayısı ciddi şekilde azalıyor ve
# ağaçlar daha küçük, model dosyası daha hafif oluyor.
# ============================================================
MIN_SAYIM = 20

marka_sayilari = df["Marka"].value_counts()
nadir_markalar = marka_sayilari[marka_sayilari < MIN_SAYIM].index
df["Marka"] = df["Marka"].where(~df["Marka"].isin(nadir_markalar), "Diger")

tip_sayilari = df["Arac_Tip"].value_counts()
nadir_tipler = tip_sayilari[tip_sayilari < MIN_SAYIM].index
df["Arac_Tip"] = df["Arac_Tip"].where(~df["Arac_Tip"].isin(nadir_tipler), "Diger")

print("Gruplama sonrası benzersiz Marka sayısı:", df["Marka"].nunique())
print("Gruplama sonrası benzersiz Arac_Tip sayısı:", df["Arac_Tip"].nunique())

# ============================================================
# 6) ÖZELLİK / HEDEF AYRIMI
# ============================================================
features = [
    "Marka",
    "Arac_Tip_Grubu",
    "Arac_Tip",
    "Model_Yil",
    "Yakit_Turu",
    "Vites",
    "Kasa_Tipi",
    "Kimden",
    "Durum",
    "Km",
    "Arac_Yasi",
    "Km_Yil"
]

X = df[features]
y = df["Fiyat"]

X = pd.get_dummies(X)

print("One-hot encoding sonrası sütun sayısı:", X.shape[1])

# ============================================================
# 7) EĞİTİM / TEST AYRIMI
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ============================================================
# 8) MODEL
# ------------------------------------------------------------
# n_estimators ve max_depth düşürüldü, max_features="sqrt"
# eklendi ve min_samples_leaf artırıldı. Bu ayarlar hem
# dosya boyutunu ciddi şekilde küçültüyor hem de aşırı
# öğrenme (overfitting) riskini azaltıyor.
# ============================================================
model = RandomForestRegressor(
    n_estimators=50,
    max_depth=10,
    min_samples_leaf=10,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

print("Model eğitiliyor...")
model.fit(X_train, y_train)
print("Model eğitildi.")

# ============================================================
# 9) PERFORMANS DEĞERLENDİRME
# ============================================================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("=" * 40)
print("MODEL PERFORMANSI")
print("=" * 40)
print(f"MAE  : {mae:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"R²   : {r2:.4f}")

importance = pd.Series(
    model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print("\nEn önemli 15 değişken:\n")
print(importance.head(15))

# ============================================================
# 10) MODELİ KAYDET (maksimum sıkıştırma ile)
# ------------------------------------------------------------
# compress=("xz", 9) joblib'in en yüksek sıkıştırma seviyesi.
# compress=3'e göre dosya boyutunda ciddi bir azalma sağlar.
# ============================================================
joblib.dump(
    model,
    "arac_fiyat_modeli.pkl",
    compress=("xz", 9)
)

joblib.dump(
    X.columns.tolist(),
    "feature_columns.pkl"
)

df.to_csv(
    "temiz_veri.csv",
    index=False
)

print("\nDosyalar oluşturuldu.")
print("✔ arac_fiyat_modeli.pkl")
print("✔ feature_columns.pkl")
print("✔ temiz_veri.csv")

import os
boyut_mb = os.path.getsize("arac_fiyat_modeli.pkl") / 1e6
print(f"\narac_fiyat_modeli.pkl boyutu: {boyut_mb:.2f} MB")