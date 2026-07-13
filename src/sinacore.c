/**
 * @file sinacore.c
 * @brief Implementación unificada del backend de SINA-VISUAL.
 *        Fusiona neural_network.c y dataset.c en un solo módulo.
 */
#include "sinacore.h"
#include <ctype.h>
#include <math.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <commdlg.h>
int open_file_dialog(char *out_path, int max_len) {
    HMODULE hComdlg = LoadLibraryA("comdlg32.dll");
    if (!hComdlg) return 0;
    typedef BOOL (WINAPI *GOFN)(LPOPENFILENAMEA);
    GOFN pGetOpenFileNameA = (GOFN)GetProcAddress(hComdlg, "GetOpenFileNameA");
    if (!pGetOpenFileNameA) { FreeLibrary(hComdlg); return 0; }

    OPENFILENAMEA ofn;
    ZeroMemory(&ofn, sizeof(ofn));
    ZeroMemory(out_path, max_len);
    ofn.lStructSize  = sizeof(ofn);
    ofn.hwndOwner    = NULL;
    ofn.lpstrFilter  = "Imagenes (*.png;*.jpg;*.jpeg;*.bmp)\0*.png;*.jpg;*.jpeg;*.bmp\0Todos los archivos (*.*)\0*.*\0";
    ofn.lpstrFile    = out_path;
    ofn.nMaxFile     = max_len;
    ofn.lpstrTitle   = "Seleccionar Radiografia";
    ofn.Flags        = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
    
    int result = pGetOpenFileNameA(&ofn) ? 1 : 0;
    FreeLibrary(hComdlg);
    return result;
}
#else
int open_file_dialog(char *out_path, int max_len) {
    (void)out_path; (void)max_len; return 0;
}
#endif

/* =========================================================================
   AUXILIARES INTERNOS
   ========================================================================= */

static void trim(char *str) {
    char *start = str;
    while (*start && isspace((unsigned char)*start)) start++;
    char *end = start + strlen(start) - 1;
    while (end > start && isspace((unsigned char)*end)) end--;
    *(end + 1) = '\0';
    if (start != str) memmove(str, start, strlen(start) + 1);
}

/* =========================================================================
   RED NEURONAL
   ========================================================================= */

RedNeuronal *inicializar_red(int num_capas, int *neuronas_por_capa, double tasa_aprendizaje) {
    if (num_capas < 2 || neuronas_por_capa == NULL) return NULL;

    RedNeuronal *red = (RedNeuronal *)malloc(sizeof(RedNeuronal));
    if (!red) return NULL;

    red->num_capas = num_capas;
    red->tasa_aprendizaje = tasa_aprendizaje;
    red->neuronas_por_capa = (int *)malloc(sizeof(int) * num_capas);
    if (!red->neuronas_por_capa) { free(red); return NULL; }

    for (int i = 0; i < num_capas; i++)
        red->neuronas_por_capa[i] = neuronas_por_capa[i];

    red->capas = (Capa *)malloc(sizeof(Capa) * num_capas);
    if (!red->capas) { free(red->neuronas_por_capa); free(red); return NULL; }

    for (int i = 0; i < num_capas; i++) {
        int n = neuronas_por_capa[i];
        red->capas[i].num_neuronas = n;
        red->capas[i].neuronas = (Neurona *)malloc(sizeof(Neurona) * n);
        if (!red->capas[i].neuronas) { liberar_red(red); return NULL; }

        for (int j = 0; j < n; j++) {
            red->capas[i].neuronas[j].salida = 0.0;
            red->capas[i].neuronas[j].error  = 0.0;
            if (i == 0) {
                red->capas[i].neuronas[j].num_entradas = 0;
                red->capas[i].neuronas[j].pesos = NULL;
                red->capas[i].neuronas[j].bias  = 0.0;
            } else {
                int ni = neuronas_por_capa[i - 1];
                red->capas[i].neuronas[j].num_entradas = ni;
                red->capas[i].neuronas[j].pesos = (double *)malloc(sizeof(double) * ni);
                if (!red->capas[i].neuronas[j].pesos) { liberar_red(red); return NULL; }
                for (int k = 0; k < ni; k++)
                    red->capas[i].neuronas[j].pesos[k] = ((double)rand() / RAND_MAX) - 0.5;
                red->capas[i].neuronas[j].bias = ((double)rand() / RAND_MAX) - 0.5;
            }
        }
    }
    return red;
}

void forward_propagation(RedNeuronal *red, double *entradas) {
    if (!red || !entradas) return;
    int ni = red->capas[0].num_neuronas;
    for (int j = 0; j < ni; j++)
        red->capas[0].neuronas[j].salida = entradas[j];

    for (int i = 1; i < red->num_capas; i++) {
        int n = red->capas[i].num_neuronas;
        for (int j = 0; j < n; j++) {
            double suma = red->capas[i].neuronas[j].bias;
            int ne = red->capas[i].neuronas[j].num_entradas;
            for (int k = 0; k < ne; k++)
                suma += red->capas[i].neuronas[j].pesos[k] * red->capas[i-1].neuronas[k].salida;
            red->capas[i].neuronas[j].salida = sigmoid(suma);
        }
    }
}

void backpropagation(RedNeuronal *red, double *salidas_esperadas) {
    if (!red || !salidas_esperadas) return;
    int L = red->num_capas - 1;

    /* Gradiente capa salida */
    int no = red->capas[L].num_neuronas;
    for (int j = 0; j < no; j++) {
        double y = red->capas[L].neuronas[j].salida;
        red->capas[L].neuronas[j].error = (salidas_esperadas[j] - y) * sigmoid_derivada(y);
    }

    /* Gradientes capas ocultas */
    for (int i = L - 1; i >= 1; i--) {
        int n  = red->capas[i].num_neuronas;
        int nn = red->capas[i+1].num_neuronas;
        for (int j = 0; j < n; j++) {
            double y = red->capas[i].neuronas[j].salida;
            double sg = 0.0;
            for (int m = 0; m < nn; m++)
                sg += red->capas[i+1].neuronas[m].error * red->capas[i+1].neuronas[m].pesos[j];
            red->capas[i].neuronas[j].error = sg * sigmoid_derivada(y);
        }
    }

    /* Actualizar pesos y bias */
    for (int i = 1; i <= L; i++) {
        int n = red->capas[i].num_neuronas;
        for (int j = 0; j < n; j++) {
            double g = red->capas[i].neuronas[j].error;
            red->capas[i].neuronas[j].bias += red->tasa_aprendizaje * g;
            int ne = red->capas[i].neuronas[j].num_entradas;
            for (int k = 0; k < ne; k++)
                red->capas[i].neuronas[j].pesos[k] +=
                    red->tasa_aprendizaje * g * red->capas[i-1].neuronas[k].salida;
        }
    }
}

void liberar_red(RedNeuronal *red) {
    if (!red) return;
    if (red->capas) {
        for (int i = 0; i < red->num_capas; i++) {
            if (red->capas[i].neuronas) {
                for (int j = 0; j < red->capas[i].num_neuronas; j++)
                    if (red->capas[i].neuronas[j].pesos)
                        free(red->capas[i].neuronas[j].pesos);
                free(red->capas[i].neuronas);
            }
        }
        free(red->capas);
    }
    if (red->neuronas_por_capa) free(red->neuronas_por_capa);
    free(red);
}

double sigmoid(double x)          { return 1.0 / (1.0 + exp(-x)); }
double sigmoid_derivada(double y)  { return y * (1.0 - y); }
double relu(double x)              { return x > 0.0 ? x : 0.0; }
double relu_derivada(double y)     { return y > 0.0 ? 1.0 : 0.0; }

int guardar_pesos(RedNeuronal *red, const char *filename) {
    if (!red || !filename) return 0;
    FILE *f = fopen(filename, "wb");
    if (!f) return 0;

    fwrite(&red->num_capas, sizeof(int), 1, f);
    fwrite(red->neuronas_por_capa, sizeof(int), red->num_capas, f);

    for (int i = 1; i < red->num_capas; i++) {
        int n = red->capas[i].num_neuronas;
        for (int j = 0; j < n; j++) {
            fwrite(&red->capas[i].neuronas[j].bias, sizeof(double), 1, f);
            int ne = red->capas[i].neuronas[j].num_entradas;
            fwrite(red->capas[i].neuronas[j].pesos, sizeof(double), ne, f);
        }
    }
    fclose(f);
    return 1;
}

int cargar_pesos(RedNeuronal *red, const char *filename) {
    if (!red || !filename) return 0;
    FILE *f = fopen(filename, "rb");
    if (!f) return 0;

    int tmp_num_capas = 0;
    if (fread(&tmp_num_capas, sizeof(int), 1, f) != 1) { fclose(f); return 0; }
    if (tmp_num_capas != red->num_capas) { fclose(f); return 0; }

    int *tmp_npc = (int *)malloc(sizeof(int) * tmp_num_capas);
    if (!tmp_npc) { fclose(f); return 0; }
    if (fread(tmp_npc, sizeof(int), tmp_num_capas, f) != (size_t)tmp_num_capas) { free(tmp_npc); fclose(f); return 0; }

    for (int i = 0; i < tmp_num_capas; i++) {
        if (tmp_npc[i] != red->neuronas_por_capa[i]) { free(tmp_npc); fclose(f); return 0; }
    }
    free(tmp_npc);

    for (int i = 1; i < red->num_capas; i++) {
        int n = red->capas[i].num_neuronas;
        for (int j = 0; j < n; j++) {
            if (fread(&red->capas[i].neuronas[j].bias, sizeof(double), 1, f) != 1) { fclose(f); return 0; }
            int ne = red->capas[i].neuronas[j].num_entradas;
            if (fread(red->capas[i].neuronas[j].pesos, sizeof(double), ne, f) != (size_t)ne) { fclose(f); return 0; }
        }
    }
    fclose(f);
    return 1;
}

/* =========================================================================
   CONFIGURACIÓN Y DATASET
   ========================================================================= */

Config *cargar_configuracion(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) { perror("Error al abrir config"); return NULL; }

    Config *cfg = (Config *)malloc(sizeof(Config));
    if (!cfg) { fclose(file); return NULL; }

    cfg->num_capas = 0;
    cfg->neuronas_por_capa = NULL;
    cfg->tasa_aprendizaje = 0.01;
    cfg->epocas = 100;
    cfg->dataset_path[0] = '\0';

    char line[512];

    /* Primera pasada: obtener NUM_CAPAS */
    while (fgets(line, sizeof(line), file)) {
        char *c = strchr(line, '#'); if (c) *c = '\0';
        trim(line);
        if (!strlen(line)) continue;
        char *eq = strchr(line, '=');
        if (eq) {
            *eq = '\0';
            char key[256], val[256];
            strcpy(key, line); strcpy(val, eq + 1);
            trim(key); trim(val);
            if (strcmp(key, "NUM_CAPAS") == 0) {
                cfg->num_capas = atoi(val);
                if (cfg->num_capas > 0)
                    cfg->neuronas_por_capa = (int *)calloc(cfg->num_capas, sizeof(int));
            }
        }
    }

    if (cfg->num_capas <= 0 || !cfg->neuronas_por_capa) {
        free(cfg); fclose(file); return NULL;
    }

    /* Segunda pasada: resto de parámetros */
    rewind(file);
    while (fgets(line, sizeof(line), file)) {
        char *c = strchr(line, '#'); if (c) *c = '\0';
        trim(line);
        if (!strlen(line)) continue;
        char *eq = strchr(line, '=');
        if (eq) {
            *eq = '\0';
            char key[256], val[256];
            strcpy(key, line); strcpy(val, eq + 1);
            trim(key); trim(val);
            if      (strcmp(key, "TASA_APRENDIZAJE") == 0) cfg->tasa_aprendizaje = atof(val);
            else if (strcmp(key, "EPOCAS") == 0)           cfg->epocas = atoi(val);
            else if (strcmp(key, "DATASET") == 0)          strcpy(cfg->dataset_path, val);
            else if (strncmp(key, "NEURONAS_CAPA", 13) == 0) {
                int idx = atoi(key + 13) - 1;
                if (idx >= 0 && idx < cfg->num_capas)
                    cfg->neuronas_por_capa[idx] = atoi(val);
            }
        }
    }
    fclose(file);
    return cfg;
}

void liberar_config(Config *cfg) {
    if (!cfg) return;
    if (cfg->neuronas_por_capa) free(cfg->neuronas_por_capa);
    free(cfg);
}

Dataset *cargar_dataset(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) { perror("Error al abrir dataset"); return NULL; }

    Dataset *ds = (Dataset *)malloc(sizeof(Dataset));
    if (!ds) { fclose(file); return NULL; }

    ds->num_samples = ds->num_features = 0;
    ds->inputs = NULL; ds->targets = NULL;

    char *line = (char*)malloc(65536);
    if (!line) { free(ds); fclose(file); return NULL; }
    int first = 1;

    while (fgets(line, 65536, file)) {
        char *cm = strchr(line, '#'); if (cm) *cm = '\0';
        trim(line);
        if (!strlen(line)) continue;
        if (first) {
            int commas = 0;
            for (int i = 0; line[i]; i++) if (line[i] == ',') commas++;
            if (commas < 1) { free(line); free(ds); fclose(file); return NULL; }
            ds->num_features = commas;
            first = 0;
        }
        ds->num_samples++;
    }

    if (!ds->num_samples || !ds->num_features) { free(line); free(ds); fclose(file); return NULL; }

    ds->inputs  = (double **)malloc(sizeof(double *) * ds->num_samples);
    ds->targets = (double  *)malloc(sizeof(double)   * ds->num_samples);
    if (!ds->inputs || !ds->targets) { free(line); liberar_dataset(ds); fclose(file); return NULL; }

    for (int i = 0; i < ds->num_samples; i++) {
        ds->inputs[i] = (double *)malloc(sizeof(double) * ds->num_features);
        if (!ds->inputs[i]) { free(line); liberar_dataset(ds); fclose(file); return NULL; }
    }

    rewind(file);
    int si = 0;
    while (fgets(line, 65536, file) && si < ds->num_samples) {
        char *cm = strchr(line, '#'); if (cm) *cm = '\0';
        trim(line);
        if (!strlen(line)) continue;
        char *tok = strtok(line, ",");
        int fi = 0;
        while (tok) {
            trim(tok);
            if      (fi < ds->num_features) ds->inputs[si][fi]  = atof(tok);
            else if (fi == ds->num_features) ds->targets[si]     = atof(tok);
            tok = strtok(NULL, ",");
            fi++;
        }
        si++;
    }
    free(line);
    fclose(file);
    return ds;
}

void liberar_dataset(Dataset *ds) {
    if (!ds) return;
    if (ds->inputs) {
        for (int i = 0; i < ds->num_samples; i++)
            if (ds->inputs[i]) free(ds->inputs[i]);
        free(ds->inputs);
    }
    if (ds->targets) free(ds->targets);
    free(ds);
}
