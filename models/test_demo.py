# ATHÉNA - models/test_demo.py
# Vérifie que le scénario de démo guidée se déroule de façon déterministe et produit un PDF valide.
# Lancement : python -m models.test_demo  (depuis la racine du projet)
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analytics.scenario_demo import construire_scenario, generer_pdf_demo  # noqa: E402


def main():
    ok = True

    def verif(nom, cond, detail=""):
        nonlocal ok
        print(f"[{'OK' if cond else 'ÉCHEC'}] {nom} {detail}")
        ok &= bool(cond)

    a, b = construire_scenario(), construire_scenario()
    verif("scénario déterministe (2 exécutions identiques)",
          a["profil_risque"]["injury_risk_pct"] == b["profil_risque"]["injury_risk_pct"])
    verif("risque élevé (> 66 %)", a["profil_risque"]["injury_risk_pct"] > 66, f"({a['profil_risque']['injury_risk_pct']} %)")
    verif("ACWR en zone 'Risque élevé'", a["acwr_info"]["zone"] == "Risque élevé", f"({a['acwr_info']['acwr']})")
    verif("what-if sommeil : le risque baisse", a["what_if_sommeil"]["apres_pct"] < a["what_if_sommeil"]["avant_pct"])
    verif("plan combiné : le risque baisse", a["plan_action"]["plan_combine"]["gain_pts"] > 0)
    verif("au moins une alerte critique", any(x[0] == "Critique" for x in a["alertes"]))
    with tempfile.TemporaryDirectory() as d:
        chemin = generer_pdf_demo(a, str(Path(d) / "demo.pdf"))
        taille = Path(chemin).stat().st_size
        verif("PDF généré", taille > 3000, f"({taille} octets)")
        verif("PDF valide", Path(chemin).read_bytes()[:5] == b"%PDF-")
    print("\nTous les tests passent." if ok else "\nDes tests ont échoué.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
