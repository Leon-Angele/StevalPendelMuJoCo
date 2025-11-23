import mujoco
import os

# Geben Sie den Pfad zu Ihrer URDF-Datei an
urdf_path = 'pendel_roboter.urdf'

# Geben Sie den Pfad für die Ausgabe der MJCF-Datei an
mjcf_path = 'pendel_roboter.xml'

try:
    # 1. URDF-Datei laden. MuJoCo konvertiert sie intern in ein MJCF-Modell.
    # Hier wird ein MjModel-Objekt erstellt
    model = mujoco.MjModel.from_xml_path(urdf_path)
    
    # 2. Das zuletzt geladene XML (jetzt im MJCF-Format) speichern.
    # MuJoCo speichert das intern generierte MJCF-Äquivalent.
    mujoco.mj_saveLastXML(mjcf_path, model)
    
    print(f"✅ Erfolgreich konvertiert und als '{mjcf_path}' gespeichert.")
    
except Exception as e:
    print(f"❌ Fehler bei der Konvertierung: {e}")
    print("Stellen Sie sicher, dass die URDF-Datei gültig ist und alle referenzierten Dateien (Meshes/Assets) gefunden werden können.")