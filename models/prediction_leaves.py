import pandas as pd
import os

# =============================
# SAFE LOADING FUNCTION
# =============================
def load_csv_safe(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Fichier manquant : {file_path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Fichier chargé : {file_path} ({df.shape[0]} lignes, {df.shape[1]} colonnes)")
        return df
    except Exception as e:
        print(f"❌ Erreur de lecture du fichier {file_path} : {e}")
        return pd.DataFrame()


# =============================
# EMPLOYEES WITH HIGH RISK (>=5 DEMANDS)
# =============================
def high_risk_employees():
    demandes = load_csv_safe("Demandes_Conges.csv")
    if demandes.empty:
        print("❌ Impossible de continuer, fichier vide.")
        return

    # Compter le nombre de demandes par employé
    demand_counts = demandes.groupby("Demandeur").size().reset_index(name="Number_of_demands")

    # Filtrer ceux avec >=5 demandes
    high_risk = demand_counts[demand_counts["Number_of_demands"] >= 5].copy()

    # Ajouter le risque avec emoji
    high_risk["Risque"] = "⚠️ Risque élevé"

    # Trier par nombre de demandes décroissant
    high_risk = high_risk.sort_values("Number_of_demands", ascending=False)

    print("\n===== Employés à risque élevé (>=5 demandes) =====\n")
    print(high_risk)

    # Sauvegarde en CSV
    high_risk.to_csv("High_Risk_Employees.csv", index=False)
    print("\n📊 Liste sauvegardée : High_Risk_Employees.csv")


# =============================
# MAIN
# =============================
if __name__ == "__main__":
    high_risk_employees()
