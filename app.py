"""
SPK Seleksi Penerima Beasiswa - Metode TOPSIS
Jalankan dengan:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SPK Beasiswa - TOPSIS", page_icon="🎓", layout="wide")


# ----------------------------------------------------------------------
# Fungsi inti metode TOPSIS
# ----------------------------------------------------------------------
def hitung_topsis(df_nilai: pd.DataFrame, bobot: dict, jenis: dict):
    """
    df_nilai : kolom 'Nama' + kolom kriteria (angka > 0)
    bobot    : {kriteria: bobot}, total = 1
    jenis    : {kriteria: 'Benefit' / 'Cost'}
    Return   : dict berisi R, Y, A+, A-, dan tabel hasil
    """
    kriteria = list(bobot.keys())
    nilai = df_nilai.set_index("Nama")[kriteria].astype(float)
    w = np.array([bobot[k] for k in kriteria])
    is_cost = np.array([jenis[k] == "Cost" for k in kriteria])

    X = nilai.values
    R = X / np.sqrt((X ** 2).sum(axis=0))          # normalisasi vektor
    Y = R * w                                       # normalisasi terbobot

    a_plus = np.where(is_cost, Y.min(axis=0), Y.max(axis=0))
    a_minus = np.where(is_cost, Y.max(axis=0), Y.min(axis=0))

    d_plus = np.sqrt(((Y - a_plus) ** 2).sum(axis=1))
    d_minus = np.sqrt(((Y - a_minus) ** 2).sum(axis=1))
    v = d_minus / (d_plus + d_minus)

    hasil = pd.DataFrame(
        {"D+": d_plus, "D-": d_minus, "Nilai Preferensi (V)": v}, index=nilai.index
    ).sort_values("Nilai Preferensi (V)", ascending=False)
    hasil.insert(0, "Peringkat", range(1, len(hasil) + 1))

    return {
        "R": pd.DataFrame(R, index=nilai.index, columns=kriteria),
        "Y": pd.DataFrame(Y, index=nilai.index, columns=kriteria),
        "ideal": pd.DataFrame({"A+ (ideal positif)": a_plus, "A- (ideal negatif)": a_minus}, index=kriteria),
        "hasil": hasil,
    }


# ----------------------------------------------------------------------
# Data contoh
# ----------------------------------------------------------------------
KRITERIA_DEFAULT = pd.DataFrame(
    {
        "Kriteria": ["IPK", "Penghasilan Orang Tua", "Jumlah Tanggungan", "Prestasi Non-Akademik"],
        "Bobot": [35.0, 25.0, 15.0, 25.0],
        "Jenis": ["Benefit", "Cost", "Benefit", "Benefit"],
    }
)

DATA_DEFAULT = pd.DataFrame(
    {
        "Nama": ["Rina", "Dimas", "Salsa", "Fajar", "Putri"],
        "IPK": [3.80, 3.60, 3.90, 3.50, 3.70],
        "Penghasilan Orang Tua": [4.5, 2.5, 6.0, 2.0, 3.5],
        "Jumlah Tanggungan": [3, 4, 2, 5, 3],
        "Prestasi Non-Akademik": [80, 75, 90, 60, 85],
    }
)

# ----------------------------------------------------------------------
# Tampilan
# ----------------------------------------------------------------------
st.title("🎓 SPK Seleksi Penerima Beasiswa")
st.caption("Metode TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)")

with st.expander("📘 Tentang kasus & metode"):
    st.markdown(
        """
**Masalah:** seleksi beasiswa manual lambat, rawan salah, dan sulit memadukan banyak kriteria.  
**Keputusan:** mahasiswa mana yang layak menerima beasiswa sesuai kuota.

**Langkah TOPSIS:**
1. Normalisasi: `r = x / √Σx²`
2. Normalisasi terbobot: `y = w × r`
3. Solusi ideal A⁺ dan A⁻ (benefit: maks/min, cost: min/maks)
4. Jarak: `D⁺` dan `D⁻`
5. Nilai preferensi: `V = D⁻ / (D⁻ + D⁺)`
        """
    )

tab1, tab2, tab3 = st.tabs(["1️⃣ Kriteria & Bobot", "2️⃣ Data Mahasiswa", "3️⃣ Hasil & Keputusan"])

# ---------------- Tab 1 ----------------
with tab1:
    st.subheader("Kriteria penilaian")
    st.write("Atur nama, bobot (%), dan jenis kriteria. Bobot dinormalisasi otomatis menjadi total 100%.")
    kriteria_df = st.data_editor(
        st.session_state.get("kriteria_df", KRITERIA_DEFAULT),
        num_rows="dynamic",
        width="stretch",
        column_config={
            "Bobot": st.column_config.NumberColumn("Bobot (%)", min_value=0.0, step=1.0),
            "Jenis": st.column_config.SelectboxColumn("Jenis", options=["Benefit", "Cost"]),
        },
        key="editor_kriteria",
    )
    st.session_state["kriteria_df"] = kriteria_df

    kriteria_df = kriteria_df.dropna(subset=["Kriteria"])
    kriteria_df = kriteria_df[kriteria_df["Kriteria"].astype(str).str.strip() != ""]
    total_bobot = kriteria_df["Bobot"].fillna(0).sum()
    if total_bobot > 0:
        st.info(f"Total bobot input: **{total_bobot:.1f}%** (dinormalisasi menjadi 100%).")
    else:
        st.warning("Total bobot masih 0.")

# ---------------- Tab 2 ----------------
with tab2:
    st.subheader("Nilai mahasiswa")
    daftar_kriteria = kriteria_df["Kriteria"].astype(str).tolist()

    sumber = st.radio("Sumber data", ["Contoh data", "Input manual", "Unggah CSV"], horizontal=True)

    if sumber == "Unggah CSV":
        st.caption("Format CSV: kolom `Nama`, lalu satu kolom untuk tiap kriteria.")
        file = st.file_uploader("Pilih file CSV", type="csv")
        data_awal = pd.read_csv(file) if file is not None else pd.DataFrame(columns=["Nama"] + daftar_kriteria)
    elif sumber == "Contoh data":
        data_awal = DATA_DEFAULT.copy()
    else:
        data_awal = pd.DataFrame({"Nama": [""] * 3})
        for k in daftar_kriteria:
            data_awal[k] = 0.0

    for k in daftar_kriteria:
        if k not in data_awal.columns:
            data_awal[k] = 0.0
    data_awal = data_awal[["Nama"] + daftar_kriteria]

    data_df = st.data_editor(
        data_awal,
        num_rows="dynamic",
        width="stretch",
        key=f"editor_data_{sumber}_{'_'.join(daftar_kriteria)}",
    )
    st.session_state["data_df"] = data_df

# ---------------- Tab 3 ----------------
with tab3:
    st.subheader("Hasil perhitungan TOPSIS")
    data_df = st.session_state.get("data_df")

    error = None
    if total_bobot <= 0:
        error = "Bobot kriteria belum diisi."
    elif data_df is None or data_df.empty:
        error = "Data mahasiswa belum diisi."
    else:
        data_bersih = data_df.dropna(subset=["Nama"])
        data_bersih = data_bersih[data_bersih["Nama"].astype(str).str.strip() != ""]
        if len(data_bersih) < 2:
            error = "Minimal 2 mahasiswa diperlukan."
        elif data_bersih[daftar_kriteria].isna().any().any():
            error = "Ada nilai kosong pada data mahasiswa."
        elif (data_bersih[daftar_kriteria].astype(float) <= 0).any().any():
            error = "Semua nilai harus lebih dari 0."
        elif data_bersih["Nama"].duplicated().any():
            error = "Nama mahasiswa harus unik."

    if error:
        st.warning(error)
    else:
        bobot = {r["Kriteria"]: float(r["Bobot"] or 0) / total_bobot for _, r in kriteria_df.iterrows()}
        jenis = {r["Kriteria"]: r["Jenis"] for _, r in kriteria_df.iterrows()}

        out = hitung_topsis(data_bersih, bobot, jenis)
        hasil = out["hasil"]

        kuota = st.number_input("Kuota penerima beasiswa", min_value=1, max_value=len(hasil), value=min(2, len(hasil)), step=1)
        hasil["Keputusan"] = ["✅ Direkomendasikan" if i < kuota else "—" for i in range(len(hasil))]

        penerima = hasil.index[:kuota].tolist()
        c1, c2 = st.columns(2)
        c1.metric("Peringkat 1", hasil.index[0])
        c2.metric("Nilai preferensi tertinggi", f"{hasil.iloc[0]['Nilai Preferensi (V)']:.4f}")
        st.success("✅ Rekomendasi penerima beasiswa: **" + ", ".join(penerima) + "**")

        st.markdown("#### Peringkat")
        st.dataframe(
            hasil.style.format({"D+": "{:.4f}", "D-": "{:.4f}", "Nilai Preferensi (V)": "{:.4f}"}),
            width="stretch",
        )
        st.bar_chart(hasil["Nilai Preferensi (V)"])

        with st.expander("Lihat langkah perhitungan"):
            st.markdown("**Bobot ternormalisasi & jenis kriteria**")
            st.dataframe(pd.DataFrame({"Bobot": bobot, "Jenis": jenis}).style.format({"Bobot": "{:.3f}"}), width="stretch")
            st.markdown("**Matriks keputusan (X)**")
            st.dataframe(data_bersih.set_index("Nama"), width="stretch")
            st.markdown("**Matriks ternormalisasi (R)**")
            st.dataframe(out["R"].style.format("{:.4f}"), width="stretch")
            st.markdown("**Matriks ternormalisasi terbobot (Y)**")
            st.dataframe(out["Y"].style.format("{:.4f}"), width="stretch")
            st.markdown("**Solusi ideal**")
            st.dataframe(out["ideal"].style.format("{:.4f}"), width="stretch")

        st.download_button(
            "⬇️ Unduh hasil (CSV)",
            hasil.reset_index().rename(columns={"index": "Nama"}).to_csv(index=False).encode("utf-8"),
            file_name="hasil_ranking_topsis.csv",
            mime="text/csv",
        )
