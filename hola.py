# ARRIBA DEL TODO:
import os
import sys
import gc
import time
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter
import multiprocessing as mp

# --- CONFIG: OJO CON ESTA RUTA ---
INPUT_FILE = r"C:\Users\Wposs\Downloads\Cristofer\SeparadorVozBanda\entrada\cancion.mp3"  # <-- AJÚSTALA
OUTPUT_DIR = r"C:\Users\Wposs\Downloads\Cristofer\SeparadorVozBanda\salidas"              # puede ser absoluta o relativa

def main():
    # 1) Resolver archivo de entrada (argumento CLI o constante)
    input_file = sys.argv[1] if len(sys.argv) > 1 else INPUT_FILE
    input_file = os.path.abspath(input_file)

    if not os.path.isfile(input_file):
        raise FileNotFoundError(
            f"Archivo de entrada no encontrado:\n  {input_file}\n\n"
            "Verifica el nombre/extensión y que la ruta sea correcta.\n"
            "Sugerencia: arrastra el archivo a la consola para pegar la ruta."
        )

    # 2) Preparar carpeta de salida (soporta ruta relativa o absoluta)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_root = OUTPUT_DIR if os.path.isabs(OUTPUT_DIR) else os.path.join(base_dir, OUTPUT_DIR)
    os.makedirs(out_root, exist_ok=True)

    # Subcarpeta por nombre de archivo para no sobrescribir entre ejecuciones
    name = os.path.splitext(os.path.basename(input_file))[0]
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)

    # 3) Cargar audio y separar
    audio_loader = AudioAdapter.default()
    sample_rate = 44100
    waveform, _ = audio_loader.load(input_file, sample_rate=sample_rate)

    separator = Separator('spleeter:2stems')  # vocals + accompaniment
    prediction = separator.separate(waveform)

    # 4) Guardar resultados
    vocals = prediction['vocals']
    accompaniment = prediction['accompaniment']
    vocals_out = os.path.join(out_dir, "vocals.wav")
    music_out = os.path.join(out_dir, "accompaniment.wav")
    audio_loader.save(vocals_out, vocals, sample_rate)
    audio_loader.save(music_out, accompaniment, sample_rate)

    print("✅ Listo:")
    print(f"  → {vocals_out}")
    print(f"  → {music_out}")

    # 5) Limpieza opcional para suprimir warnings al cerrar
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()
    time.sleep(0.2)

if __name__ == '__main__':
    mp.freeze_support()
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
