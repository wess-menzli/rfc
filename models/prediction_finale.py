import pandas as pd
import os
import sys   # 👈 to capture arguments from PAD


# =============================
# SAFE LOADING FUNCTION
# =============================
def load_csv_safe(file_path: str) -> pd.DataFrame:
    """Safely load a CSV file with multiple separators and encodings."""
    if not os.path.exists(file_path):
        print(f"❌ Fichier manquant : {file_path}")
        return pd.DataFrame()

    try:
        for sep in [',', ';']:
            for enc in ['utf-8', 'latin1']:
                try:
                    df = pd.read_csv(file_path, sep=sep, encoding=enc)
                    if df.shape[1] > 1:
                        return df
                except Exception:
                    continue
        # Fallback
        return pd.read_csv(file_path, engine="python", encoding="latin1")
    except Exception as e:
        print(f"❌ Erreur de lecture du fichier {file_path} : {e}")
        return pd.DataFrame()


# =============================
# SOLDE FUNCTION
# =============================
def get_employee_solde(soldes: pd.DataFrame, demandeur_name: str) -> float:
    """Return solde (leave balance) for a given employee name."""
    demandeur_name = demandeur_name.strip().upper()

    for col in ["Nom_Prenom", "Employee"]:
        if col in soldes.columns:
            solde = soldes[soldes[col].astype(str).str.strip().str.upper() == demandeur_name]["Solde"]
            if not solde.empty:
                return float(solde.iloc[0])
    return 0.0


# =============================
# PREDICTION FUNCTION
# =============================

def predict_next_leave(demandeur_name: str) -> str:
    """Predict the next leave approval/rejection probability for an employee."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    demandes = load_csv_safe(r"C:\Users\HP\Desktop\prediction\models\Demandes_Conges.csv")
    soldes   = load_csv_safe(r"C:\Users\HP\Desktop\prediction\models\Solde_Conges.csv")


    if demandes.empty or soldes.empty:
        return "❌ Fichiers manquants ou vides."

    nom = demandeur_name.strip().upper()

    if "Demandeur" not in demandes.columns:
        return "❌ Colonne 'Demandeur' manquante dans demandes"

    df_emp = demandes[demandes["Demandeur"].astype(str).str.strip().str.upper() == nom]
    if df_emp.empty:
        return f"❌ Aucune demande trouvée pour {demandeur_name}"

    solde = get_employee_solde(soldes, nom)

    if "Status_globale" not in df_emp.columns:
        df_emp["Status_globale"] = "Validé"

    validées = df_emp[df_emp["Status_globale"].astype(str).str.strip().str.upper() == "VALIDÉ"].shape[0]
    rejetées = df_emp[df_emp["Status_globale"].astype(str).str.strip().str.upper() == "NON VALIDÉ"].shape[0]

    total = validées + rejetées
    prob_valid = validées / total if total > 0 else 0.7
    prob_reject = rejetées / total if total > 0 else 0.3

    # Get last request
    last_request = df_emp.sort_values("Date_debut_congé", ascending=False).iloc[0]
    requested_days = float(last_request["Totale_des_jours"])

    # Adjust probabilities if requested days exceed solde
    if requested_days > solde:
        prob_valid, prob_reject = 0.0, 1.0

    prediction = "✅ Probable Validée" if prob_valid >= prob_reject else "❌ Probable Rejetée"

    return (
        f"📌 Employé : {demandeur_name}\n"
        f"💰 Solde disponible : {solde} jours\n"
        f"📄 Dernière demande : {requested_days} jours\n"
        f"⚖️ Verdict prédit : {prediction} "
        f"(Validée {prob_valid*100:.1f}% / Rejetée {prob_reject*100:.1f}%)"
    )


# =============================
# MAIN
# =============================
if __name__ == "__main__":
    # 🔑 Get employee name from PAD argument
    demandeur = sys.argv[1] if len(sys.argv) > 1 else input("👤 Entrez le nom de l'employé : ")

    result = predict_next_leave(demandeur)

    # 👉 PAD captures this print in %PythonExecutionResult%
    print(result)
with open("python_out.txt", "w", encoding="utf-8") as f:
    f.write(result + "\n")
