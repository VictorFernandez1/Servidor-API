# Este script sirve como prueba de una futura sección de redes neuronales en la app de kivy.
# Hay que probar a prepocesar un archivo, usar la red neuronal ya entrenada y mostrar los resultados.


import pandas as pd
import logging




# Configuración de los logs


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Funciones

# Esta función de preprocesa separa un archivo en ciclos, elimina el primer ciclo y devuelve los valores característicos calculados.
def Preprocesa_TomateFermentado(archivo):
    try:
        df=archivo
        #df = pd.read_csv(archivo, delimiter=',', header=0)
        # Eliminaremos todas las columnas salvo 'Sample', 'R(BME680)', 'H2(SGP30)', 'SRAW(SGP40)', 'Ethanol(SGP30)', 'Air/Sample'
        columnas_a_eliminar = [col for col in df.columns if col not in ['Sample', 'R(BME680)', 'H2(SGP30)', 'SRAW(SGP40)', 'Ethanol(SGP30)', 'Air/Sample']]
        df = df.drop(columns=columnas_a_eliminar, errors='ignore')
        #A continuación, realizaremos una media móvil de 3 puntos a todas las columnas, salvo la primera y la última.
        cols = df.columns[1:-1]
        df[cols] = df[cols].astype(float)
        df[cols] = df[cols].rolling(window=3, min_periods=1).mean()

        # Quizá faltan algunas filas. Comprobaremos si el número de filas del dataframe es igual al último valor del la primera columna.
        if int(df.iloc[-1, 0]) != df.shape[0]:
            logging.info(f" En el archivo  {archivo} faltan {int(df.iloc[-1, 0]) - df.shape[0]} filas, serán sustituidas por la fila anterior.")
            # Veremos que filas falta en el dataframe. Para ello, recorreremos la primera columna del dataframe. Esta columna es un índice que empieza por 1 y llega hasta int(df.iloc[-1, 0])
            # Si el valor de la fila es diferente al índice, añadimos una fila con el valor del índice y el resto de columnas con el valor de la fila anterior.
            # Si el valor de la fila es igual al índice, no hacemos nada.

            for i in range(1, int(df.iloc[-1, 0]) + 1):
                if i != int(df.iloc[i - 1, 0]):
                    # Añadimos una fila con el valor del índice y el resto de columnas con el valor de la fila anterior.
                    nueva_fila = df.iloc[i - 2].copy()
                    nueva_fila.iloc[0] = i
                    df = pd.concat([df.iloc[:i - 1], nueva_fila.to_frame().T, df.iloc[i - 1:]]).reset_index(drop=True)


        #Añadiremos una columna que indique el número de ciclo. Para ello, contaremos el número de veces que aparece un 0 en la columna "Air/Sample" hasta que cambia a 1.
        ceros = 0
        unos = 0
        for i in range(df.shape[0]):
            if float(df.iloc[i]["Air/Sample"]) == 1:
                unos += 1
            else:
                for j in range(df.shape[0]):
                    if float(df.iloc[j + unos + 1]["Air/Sample"]) == 0:
                        ceros +=1
                    else:
                        break
                break
        ceros += 1      # Añadimos uno porque el último no lo cuenta, se puede mejorar el código pero me da pereza.
        #unos +=1
        tamaño_ciclo = ceros + unos

        #A continuación, calculamos el número de ciclos dividiendo el número de filas del dataframe por el tamaño de un ciclo.
        num_ciclos=int(df.shape[0]/tamaño_ciclo)

        if int(df.iloc[-1, 0]) != int(num_ciclos)*(int(tamaño_ciclo)):


            logging.info(f"It seems that the file suffered a loss of connection. The number of cycles will be recalculated.")
            #if input("Presiona S para preprocesar el archivo o cualquier otra tecla para no preprocesarlo: ") in ("S", "s"):
                #Recalculamos el número de ciclos. Dividimos el número de filas entre el número de filas por ciclo y truncamos el resultado.
            NumCiclosRecalculado = (df.shape[0] // tamaño_ciclo)
            logging.info(f"El número de ciclos es {NumCiclosRecalculado}")
            #Eliminamos las filas que sobran. Estas son aquellas que están por encima del número de filas por ciclo multiplicado por el número recalculado de ciclos.
            df = df.iloc[:(NumCiclosRecalculado*tamaño_ciclo)]
            num_ciclos=NumCiclosRecalculado
            #logging.debug(f"Último valor de la primera columna: {df.iloc[-1, 0]}")

            #else:
                #raise ValueError("El archivo sufrió pérdida de conexión y no fue preprocesado")

        # Añadir dos nuevas columnas vacías con nombre "sample_ciclo" y "sample"
        df['sample_ciclo'] = 0
        df['ciclo'] = 0

        # Una vez conocido el tamaño de ciclo. Añadimos una columna que indique el número de ciclo y otra que indique el número de sample dentro del ciclo.
        for i in range(0, num_ciclos):  # i será el número de ciclo
            # Seleccionar las filas del ciclo actual
            for j in range(0, tamaño_ciclo):
                df.loc[int(j + (i) * int(tamaño_ciclo)), 'sample_ciclo'] = j+1      #Lo de añadir ceros es para considerar que un ciclo empieza por la adsorción.
                df.loc[int(j + (i) * int(tamaño_ciclo)), 'ciclo'] = i



        # Calculamos el número de ciclos únicos en la columna 'ciclo'
        num_ciclos = df['ciclo'].nunique()

        # Separaremos el dataframe en ciclos. Para ello, crearemos una lista de dataframes, cada uno de ellos representando un ciclo.
        ciclos = []
        ciclos_valores_mínimos = []                 # Lista para almacenar los valores mínimos de cada ciclo
        ciclos_línea_base = []                      # Lista para almacenar la línea base de cada ciclo
        ciclos_delta = []                           # Lista para almacenar el delta de cada ciclo
        ciclos_delta_rel = []                       # Lista para almacenar el delta relativo(delta/línea base) de cada ciclo
        # Empezamos por el ciclo 1 , ya que el ciclo 0 no nos interesa.
        for i in range(1, num_ciclos):
            ciclo_df = df[df['ciclo'] == i].reset_index(drop=True)
            ciclos.append(ciclo_df)

        # Eliminaremos las columas 'ciclo', 'Sample' y 'Air/Sample' para cada ciclo, ya que no son necesarias en el dataframe final.
        for i in range(len(ciclos)):
            ciclos[i] = ciclos[i].drop(columns=['ciclo', 'Sample', 'Air/Sample', 'sample_ciclo','Muestra'], errors='ignore')
            #Recorremos cada columna del ciclo y guardamos el valor mínimo en un dataframe aparte.
            ciclos_valores_mínimos.append(ciclos[i].min().to_frame().T)  # Convertimos la Serie en un DataFrame de una sola fila
            #Hallamos la línea base de cada columna del ciclo y la guardamos en un dataframe aparte. La línea base es el primer valor de cada columna del ciclo.
            ciclos_línea_base.append(ciclos[i].iloc[0].to_frame().T)  # Convertimos la Serie en un DataFrame de una sola fila
            #Hallamos el delta de cada columna del ciclo y la guardamos en un dataframe aparte. El delta es la diferencia entre el valor mínimo y la línea base.
            ciclos_delta.append(ciclos_valores_mínimos[i] - ciclos_línea_base[i])  # Calculamos el delta y lo guardamos en un dataframe
            # Hallamos el delta relativo de cada columna del ciclo y lo guardamos en un dataframe aparte. El delta relativo es el delta dividido por la línea base.
            ciclos_delta_rel.append(ciclos_delta[i] / ciclos_línea_base[i])


        # Concatenamos los dataframes de ciclos en uno solo, para que sea más fácil trabajar con ellos.
        ciclos_valores_mínimos = pd.concat(ciclos_valores_mínimos, ignore_index=True)
        #Modificaremos el nombre de las columnas de ciclos_valores_mínimos. El nombre de la columna será el nombre de la columna original + "_min".
        ciclos_valores_mínimos.columns = [col + '_min' for col in ciclos_valores_mínimos.columns]

        ciclos_delta = pd.concat(ciclos_delta, ignore_index=True)
        ciclos_delta.columns = [col + '_delta' for col in ciclos_delta.columns]

        ciclos_delta_rel = pd.concat(ciclos_delta_rel, ignore_index=True)
        ciclos_delta_rel.columns = [col + '_delta_rel' for col in ciclos_delta_rel.columns]


        ciclos_valores_característicos = pd.concat([ciclos_valores_mínimos, ciclos_delta,ciclos_delta_rel], axis=1)  # Concatenamos los dataframes de valores mínimos y delta

        # Borramos las últimas filas de cada ciclo. Nos quedamos únicamente con las 90 primeras filas.
        # for i, ciclo in enumerate(ciclos):
        #     ciclos[i] = ciclo.iloc[:90]
        #     logging.info(f"Ciclo {i + 1} tiene {ciclos[i].shape[0]} filas y {ciclos[i].shape[1]} columnas.")

        #añadimos una columna con el nombre del archivo
        #ciclos_valores_característicos.insert(0, 'Muestra', [csv_filename] * len(ciclos_valores_característicos))
        #ciclos_valores_característicos.insert(1, 'Muestra_copy', [csv_filename] * len(ciclos_valores_característicos))
        print(f"Se han preprocesado {len(ciclos)} ciclos en el archivo")
        return ciclos_valores_característicos



    except Exception as e:
        logging.error(f"Error while preprocessing the file: {e}")
        # Borrar el archivo de la carpeta de archivos preprocesados en caso de error



