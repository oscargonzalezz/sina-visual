/**
 * @file sinacore.h
 * @brief Núcleo unificado de SINA-VISUAL.
 *        Contiene todas las estructuras, tipos y prototipos del backend:
 *        red neuronal, dataset y configuración.
 */
#ifndef SINACORE_H
#define SINACORE_H

#include <stdio.h>
#include <stdlib.h>

/* =========================================================================
   ESTRUCTURAS DE DATOS
   ========================================================================= */

/** Muestras de entrenamiento cargadas desde CSV. */
typedef struct {
    int num_samples;   /**< Filas del dataset */
    int num_features;  /**< Columnas de entrada por muestra */
    double **inputs;   /**< Matriz dinámica de entradas (num_samples x num_features) */
    double  *targets;  /**< Valores objetivo esperados (num_samples) */
} Dataset;

/** Una sola neurona artificial. */
typedef struct {
    double *pesos;     /**< Pesos sinápticos */
    int num_entradas;  /**< Número de entradas (tamaño de pesos[]) */
    double bias;       /**< Sesgo de la neurona */
    double salida;     /**< Último valor calculado */
    double error;      /**< Gradiente local para backpropagation */
} Neurona;

/** Capa completa de neuronas. */
typedef struct {
    Neurona *neuronas;
    int num_neuronas;
} Capa;

/** Red neuronal multicapa completa. */
typedef struct {
    Capa   *capas;
    int     num_capas;
    int    *neuronas_por_capa;
    double  tasa_aprendizaje;
} RedNeuronal;

/** Parámetros de configuración leídos desde config.txt. */
typedef struct {
    int     num_capas;
    int    *neuronas_por_capa;
    double  tasa_aprendizaje;
    int     epocas;
    char    dataset_path[256];
} Config;

/* =========================================================================
   FUNCIONES — RED NEURONAL
   ========================================================================= */
int open_file_dialog(char *out_path, int max_len);
RedNeuronal *inicializar_red(int num_capas, int *neuronas_por_capa, double tasa_aprendizaje);
void         forward_propagation(RedNeuronal *red, double *entradas);
void         backpropagation(RedNeuronal *red, double *salidas_esperadas);
void         liberar_red(RedNeuronal *red);
double       sigmoid(double x);
double       sigmoid_derivada(double x);
double       relu(double x);
double       relu_derivada(double x);
int          guardar_pesos(RedNeuronal *red, const char *filename);
int          cargar_pesos(RedNeuronal *red, const char *filename);

/* =========================================================================
   FUNCIONES — DATASET Y CONFIGURACIÓN
   ========================================================================= */
Config  *cargar_configuracion(const char *filename);
void     liberar_config(Config *config);
Dataset *cargar_dataset(const char *filename);
void     liberar_dataset(Dataset *dataset);

#endif /* SINACORE_H */
