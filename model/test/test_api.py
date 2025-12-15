import requests
import json

url = "http://127.0.0.1:8000/predict"
headers = {"Content-Type": "application/json"}

# 🔁 Liste de cas à tester
cas_de_test = [
    {
        "nom": "Cas neutre",
        "data": {
            "q606_1_avoir_faim_mais_ne_pas_manger": 0,
            "q605_1_ne_plus_avoir_de_nourriture_pas_suffisamment_d_argent": 0,
            "q604_manger_moins_que_ce_que_vous_auriez_du": 0,
            "q603_sauter_un_repas": 0,
            "q601_ne_pas_manger_nourriture_saine_nutritive": 0
        }
    },
    {
        "nom": "Cas modéré",
        "data": {
            "q606_1_avoir_faim_mais_ne_pas_manger": 1,
            "q605_1_ne_plus_avoir_de_nourriture_pas_suffisamment_d_argent": 0,
            "q604_manger_moins_que_ce_que_vous_auriez_du": 1,
            "q603_sauter_un_repas": 0,
            "q601_ne_pas_manger_nourriture_saine_nutritive": 0
        }
    },
    {
        "nom": "Cas sévère",
        "data": {
            "q606_1_avoir_faim_mais_ne_pas_manger": 1,
            "q605_1_ne_plus_avoir_de_nourriture_pas_suffisamment_d_argent": 1,
            "q604_manger_moins_que_ce_que_vous_auriez_du": 1,
            "q603_sauter_un_repas": 1,
            "q601_ne_pas_manger_nourriture_saine_nutritive": 1
        }
    }
]

# 🔁 Envoi des requêtes
for cas in cas_de_test:
    response = requests.post(url, headers=headers, data=json.dumps(cas["data"]))
    print(f"\n🧪 {cas['nom']}")
    print(response.json())
