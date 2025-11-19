import os
from PIL import Image

def preprocesar_directorio_con_recorte(directorio_entrada, directorio_salida, tamano_final=(32, 32)):
    """
    Recorre un directorio de imágenes, realiza un recorte cuadrado desde el centro,
    luego redimensiona y guarda las imágenes en otro directorio.
    """
    if not os.path.exists(directorio_salida):
        os.makedirs(directorio_salida)
        print(f"Directorio de salida creado en: {directorio_salida}")

    extensiones_validas = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    try:
        lista_archivos = [f for f in os.listdir(directorio_entrada) if f.lower().endswith(extensiones_validas)]
    except FileNotFoundError:
        print(f"Error: El directorio de entrada no fue encontrado en '{directorio_entrada}'")
        return

    if not lista_archivos:
        print(f"No se encontraron imágenes en '{directorio_entrada}'")
        return

    print(f"Se encontraron {len(lista_archivos)} imágenes. Iniciando preprocesamiento con recorte...")
    
    for i, nombre_archivo in enumerate(lista_archivos):
        ruta_completa_entrada = os.path.join(directorio_entrada, nombre_archivo)
        ruta_completa_salida = os.path.join(directorio_salida, nombre_archivo)
        
        try:
            with Image.open(ruta_completa_entrada) as img:
                # 1. Calcular las dimensiones para el recorte cuadrado
                ancho, alto = img.size
                lado_corto = min(ancho, alto)
                
                izquierda = (ancho - lado_corto) / 2
                arriba = (alto - lado_corto) / 2
                derecha = (ancho + lado_corto) / 2
                abajo = (alto + lado_corto) / 2

                # 2. Recortar la imagen desde el centro
                img_recortada = img.crop((izquierda, arriba, derecha, abajo))
                
                # 3. Redimensionar el recorte a 32x32
                img_redimensionada = img_recortada.resize(tamano_final, Image.Resampling.LANCZOS)
                
                if img_redimensionada.mode != 'RGB':
                    img_redimensionada = img_redimensionada.convert('RGB')
                    
                img_redimensionada.save(ruta_completa_salida)
                print(f"({i+1}/{len(lista_archivos)}) Procesada: {nombre_archivo}")

        except Exception as e:
            print(f"No se pudo procesar el archivo {nombre_archivo}. Error: {e}")
            
    print("\n¡Preprocesamiento con recorte completado!")
    print(f"Las imágenes procesadas se han guardado en: {directorio_salida}")


if __name__ == '__main__':
    CARPETA_ORIGINALES = r'E:\Procesamiento de Aprendizaje Automatico\Tp\pichichas'
    CARPETA_PROCESADAS = r'E:\Procesamiento de Aprendizaje Automatico\Tp\pichichas_procesadas'
    
    # Llama a la nueva función
    preprocesar_directorio_con_recorte(CARPETA_ORIGINALES, CARPETA_PROCESADAS)