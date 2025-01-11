import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

temps_inter_arrivees = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
probabilites_inter_arrivees = np.array([0.1] * 10)

temps_de_service = np.array([2, 3, 4, 5])
probabilites_service = np.array([0.25] * 4)

def generer_temps_inter_arrivee():
    return np.random.choice(temps_inter_arrivees, p=probabilites_inter_arrivees)

def generer_temps_service():
    return np.random.choice(temps_de_service, p=probabilites_service)


def simuler_file_mm1(nb_clients):
    temps_arrivees = []
    temps_debut_service = []
    temps_depart = []
    temps_actuel = 0

    for _ in range(nb_clients):
        temps_inter_arrivee = generer_temps_inter_arrivee()
        temps_arrivee = temps_actuel + temps_inter_arrivee
        temps_arrivees.append(temps_arrivee)

        if not temps_depart or temps_arrivee >= temps_depart[-1]:
            debut_service = temps_arrivee
        else:
            debut_service = temps_depart[-1]
        temps_debut_service.append(debut_service)

        temps_service = generer_temps_service()
        depart = debut_service + temps_service
        temps_depart.append(depart)

        temps_actuel = temps_arrivee

    return temps_arrivees, temps_debut_service, temps_depart


st.title("Simulation de File d'Attente M/M/1")
st.markdown("""
    <style>
        .signature {
            font-size: 24px;
            font-weight: bold;
            color: #2e3a87;
            text-align: center;
            font-family: 'Arial', sans-serif;
            padding: 20px;
            border-top: 3px solid #2e3a87;
            margin-top: 20px;
        }
    </style>
    <div class="signature">par MOHAMED ABID</div>
""", unsafe_allow_html=True)


duree_simulation = st.number_input("Durée de la simulation (unités de temps)", min_value=10.0, value=100.0, step=10.0)
nb_clients = st.number_input("Nombre de clients à simuler", min_value=1, value=100, step=1)


temps_arrivees, temps_debut_service, temps_depart = simuler_file_mm1(nb_clients)


temps_inactif = sum(max(0, temps_debut_service[i] - temps_depart[i - 1]) for i in range(1, len(temps_debut_service)))

temps_occupe = temps_depart[-1] - temps_inactif
proportion_serveur_libre = temps_inactif / temps_depart[-1]

longueurs_file = [0]
file_actuelle = 0

for i in range(1, len(temps_arrivees)):
    if temps_debut_service[i] > temps_depart[i - 1]:
        file_actuelle = 0
    else:
        file_actuelle += 1
    longueurs_file.append(file_actuelle)

longueur_moyenne_file = np.mean(longueurs_file)
longueur_moyenne_systeme = longueur_moyenne_file + temps_occupe / temps_depart[-1]

lambda_ = 1 / np.sum(temps_inter_arrivees * probabilites_inter_arrivees)
mu = 1 / np.sum(temps_de_service * probabilites_service)
rho = lambda_ / mu

Lq_analytique = (rho ** 2) / (1 - rho)
L_analytique = Lq_analytique + rho
Wq_analytique = Lq_analytique / lambda_
W_analytique = L_analytique / lambda_


st.subheader("Résultats de la Simulation")
st.write(f"Proportion de temps où le serveur est libre : {proportion_serveur_libre:.4f}")
st.write(f"Nombre moyen de clients dans la file d'attente (simulation) : {longueur_moyenne_file:.4f}")
st.write(f"Nombre moyen de clients dans le système (simulation) : {longueur_moyenne_systeme:.4f}")
st.write(f"Durée moyenne de séjour d'un client (simulation) : {longueur_moyenne_systeme / lambda_:.4f}")

st.subheader("Résultats Analytiques")
st.write(f"Nombre moyen de clients dans la file d'attente (analytique) : {Lq_analytique:.4f}")
st.write(f"Nombre moyen de clients dans le système (analytique) : {L_analytique:.4f}")
st.write(f"Durée moyenne de séjour d'un client (analytique) : {W_analytique:.4f}")
st.write(f"Proportion de temps où le serveur est libre : {1 - rho:.4f}")


points_temps = np.arange(0, duree_simulation, 0.1)
evolution_file = []
evolution_systeme = []
etat_serveur = []

for t in points_temps:
    dans_file = sum(1 for i in range(len(temps_arrivees)) if temps_arrivees[i] <= t < temps_debut_service[i])
    dans_systeme = sum(1 for i in range(len(temps_arrivees)) if temps_arrivees[i] <= t < temps_depart[i])
    evolution_file.append(dans_file)
    evolution_systeme.append(dans_systeme)
    etat_serveur.append(1 if any(temps_debut_service[i] <= t < temps_depart[i] for i in range(len(temps_arrivees))) else 0)
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(points_temps, evolution_file, label="Clients dans la file d'attente", color="blue")
ax.plot(points_temps, evolution_systeme, label="Clients dans le système", color="green")
ax.step(points_temps, etat_serveur, label="État du serveur (1=occupé, 0=libre)", color="red", where="post")

ax.set_xlabel("Temps")
ax.set_ylabel("Nombre de clients / État du serveur")
ax.set_title("Évolution du Nombre de Clients et de l'État du Serveur")
ax.legend()
ax.grid()
ax.set_xlim(0, duree_simulation)
ax.set_ylim(0, max(max(evolution_file), max(evolution_systeme), 1) + 1)

st.pyplot(fig)

 
