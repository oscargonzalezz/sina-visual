/**
 * @file app.c
 * @brief Interfaz gráfica de SINA-VISUAL — Estilo Cyber Extremo.
 *        Analizador de Radiografías con Aprendizaje Continuo.
 */
#include "sinacore.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include "raylib.h"

#ifdef _WIN32
#include <direct.h>
#define make_dir(d) _mkdir(d)
#else
#include <sys/stat.h>
#define make_dir(d) mkdir(d, 0777)
#endif


/* =========================================================================
   PALETA CYBER EXTREMA
   ========================================================================= */
#define CB_BG        (Color){0x02,0x03,0x08,0xFF}
#define CB_SIDE      (Color){0x03,0x05,0x10,0xFF}
#define CB_PANEL     (Color){0x05,0x0A,0x18,0xFF}
#define CB_PANELB    (Color){0x07,0x10,0x22,0xFF}
#define CB_CYAN      (Color){0x00,0xFF,0xF5,0xFF}
#define CB_CYAN2     (Color){0x00,0xC0,0xD8,0xFF}
#define CB_BLUE      (Color){0x20,0x80,0xFF,0xFF}
#define CB_BLUE2     (Color){0x0A,0x30,0x70,0xFF}
#define CB_PURPLE    (Color){0xB0,0x00,0xFF,0xFF}
#define CB_PURPLE2   (Color){0x60,0x00,0xA0,0xFF}
#define CB_GREEN     (Color){0x00,0xFF,0x80,0xFF}
#define CB_RED       (Color){0xFF,0x10,0x40,0xFF}
#define CB_AMBER     (Color){0xFF,0xCC,0x00,0xFF}
#define CB_TEXT      (Color){0xCC,0xEE,0xFF,0xFF}
#define CB_MUTED     (Color){0x28,0x44,0x60,0xFF}
#define CB_MUTED2    (Color){0x18,0x28,0x40,0xFF}

/* =========================================================================
   CONSTANTES Y ESTADO GLOBAL
   ========================================================================= */
#define MAX_MSG  50
#define MSG_LEN  512
#define W        1100
#define H        680
#define SIDEBAR  260

static char  chat_msg[MAX_MSG][MSG_LEN];
static int   chat_type[MAX_MSG];   /* 0=AI, 1=User, 2=Alert */
static int   chat_count = 0;

static float  *mse_hist  = NULL;
static int     mse_count = 0, mse_cap = 0;

static Config      *cfg    = NULL;
static Dataset     *ds     = NULL;
static RedNeuronal *red    = NULL;
static double       err_fin = -1.0;

static bool cfg_ok = false, ds_ok = false, red_ok = false;
static int  tab    = 0;

static bool   training    = false;
static int    epoch_cur   = 0, epoch_total = 500;
static double mse_cur     = 0.0;
static double pred_out    = -1.0;
static bool   pred_done   = false;

/* Variables para Analizador RX */
static double    rx_features[4096]; /* 64x64 matrix */
static bool      rx_loaded = false;
static Image     current_rx_img = {0};
static Texture2D current_rx_texture = {0};

/* Tiempo de animación global */
static float anim_t = 0.0f;

/* =========================================================================
   HELPERS — CHAT
   ========================================================================= */
static void add_msg(const char *text, int type) {
    if (chat_count >= MAX_MSG) {
        for (int i = 1; i < MAX_MSG; i++) {
            strcpy(chat_msg[i-1], chat_msg[i]);
            chat_type[i-1] = chat_type[i];
        }
        chat_count = MAX_MSG - 1;
    }
    strncpy(chat_msg[chat_count], text, MSG_LEN - 1);
    chat_msg[chat_count][MSG_LEN-1] = '\0';
    chat_type[chat_count] = type;
    chat_count++;
}

/* =========================================================================
   REPORTE
   ========================================================================= */
static void generar_reporte(void) {
    if (!red || !cfg) return;
    make_dir("reports");
    FILE *f = fopen("reports/final_report.txt", "w");
    if (!f) return;
    fprintf(f, "=== REPORTE SINA-VISUAL ===\n");
    fprintf(f, "Capas: %d | LR: %.4f | Epocas: %d\n",
            cfg->num_capas, cfg->tasa_aprendizaje, cfg->epocas);
    fprintf(f, "MSE Final: %.6f\n\n", err_fin);
    for (int i = 1; i < red->num_capas; i++) {
        fprintf(f, "Capa %d:\n", i+1);
        for (int j = 0; j < red->capas[i].num_neuronas; j++) {
            fprintf(f, "  N%d  bias=%.4f  pesos=[", j+1, red->capas[i].neuronas[j].bias);
            for (int k = 0; k < red->capas[i].neuronas[j].num_entradas; k++)
                fprintf(f, k ? ",%.4f" : "%.4f", red->capas[i].neuronas[j].pesos[k]);
            fprintf(f, "]\n");
        }
    }
    fclose(f);
}

static void cargar_log(void) {
    FILE *f = fopen("reports/training_log.txt", "r");
    if (!f) return;
    char line[128];
    mse_count = 0;
    while (fgets(line, sizeof(line), f)) {
        int ep; double er;
        if (sscanf(line, "%d,%lf", &ep, &er) == 2) {
            if (mse_count >= mse_cap) {
                mse_cap = mse_cap ? mse_cap*2 : 512;
                mse_hist = (float*)realloc(mse_hist, sizeof(float)*mse_cap);
            }
            mse_hist[mse_count++] = (float)er;
            epoch_cur = ep; err_fin = er;
        }
    }
    fclose(f);
}

/* =========================================================================
   EJECUTAR COMANDO
   ========================================================================= */
static void cmd(const char *raw) {
    char c[100]; int cl = 0;
    for (int i = 0; raw[i] && cl < 99; i++)
        if (!isspace((unsigned char)raw[i]))
            c[cl++] = tolower((unsigned char)raw[i]);
    c[cl] = '\0';

    add_msg(raw, 1);

    if (!strcmp(c,"ayuda")||!strcmp(c,"help")) {
        add_msg("Comandos: cargar | entrenar | grafico | analizador | reporte", 0);
    } else if (!strcmp(c,"cargar")) {
        if (cfg) { liberar_config(cfg);  cfg = NULL; }
        if (ds)  { liberar_dataset(ds);  ds  = NULL; }
        cfg = cargar_configuracion("data/config.txt");
        if (cfg) ds = cargar_dataset(cfg->dataset_path);
        if (cfg) {
            cfg_ok = true;
            if (red) { liberar_red(red); red = NULL; }
            red = inicializar_red(cfg->num_capas, cfg->neuronas_por_capa, cfg->tasa_aprendizaje);
            if (red) {
                if (cargar_pesos(red, "data/red_memoria.dat")) {
                    red_ok = true;
                    add_msg("[OK] Memoria persistente cargada con exito.", 0);
                } else {
                    add_msg("[INFO] Red inicializada. No hay memoria guardada aun.", 0);
                    guardar_pesos(red, "data/red_memoria.dat");
                    red_ok = true;
                }
            }
        }
        if (ds) ds_ok = true;
        
        if (cfg_ok) {
            char buf[256];
            snprintf(buf, sizeof(buf),
                "[OK] Config cargada. Capas:%d Entradas:%d",
                cfg->num_capas, cfg->neuronas_por_capa[0]);
            add_msg(buf, 0);
        } else {
            add_msg("[ERR] Verifica data/config.txt", 2);
        }
    } else if (!strcmp(c,"entrenar")) {
        if (!cfg_ok||!ds_ok) { add_msg("[!] Carga los datos primero.", 2); return; }
        if (red) { liberar_red(red); red = NULL; }
        red = inicializar_red(cfg->num_capas, cfg->neuronas_por_capa, cfg->tasa_aprendizaje);
        if (!red) { add_msg("[ERR] Error inicializando la red.", 2); return; }
        training = true; epoch_cur = 0; epoch_total = cfg->epocas;
        mse_count = 0; red_ok = false;
        add_msg("[BOOT] Iniciando entrenamiento...", 0);
    } else if (!strcmp(c,"grafico")||!strcmp(c,"vergrafico")) {
        tab = 2; add_msg(">> Pestaña Entrenamiento.", 0);
    } else if (!strcmp(c,"analizador")||!strcmp(c,"predecir")) {
        tab = 3; add_msg(">> Pestaña Analizador RX.", 0);
    } else if (!strcmp(c,"reporte")) {
        if (!red_ok) { add_msg("[!] Inicializa o entrena la red primero.", 2); return; }
        generar_reporte();
        char buf[128];
        snprintf(buf, sizeof(buf), "[OK] Reporte en reports/final_report.txt | MSE=%.6f", err_fin);
        add_msg(buf, 0);
    } else {
        char buf[128];
        snprintf(buf, sizeof(buf), "Comando '%s' no reconocido. Escribe 'ayuda'.", raw);
        add_msg(buf, 2);
    }
}

/* =========================================================================
   DRAW HELPERS — CYBER
   ========================================================================= */

static void GlowLine(float x0, float y0, float x1, float y1, Color col, float w) {
    DrawLineEx((Vector2){x0,y0},(Vector2){x1,y1}, w+4, (Color){col.r,col.g,col.b,20});
    DrawLineEx((Vector2){x0,y0},(Vector2){x1,y1}, w+2, (Color){col.r,col.g,col.b,50});
    DrawLineEx((Vector2){x0,y0},(Vector2){x1,y1}, w,   col);
}

static void CyberRect(Rectangle r, Color border, Color fill, int cut) {
    DrawRectangleRec(r, fill);
    DrawLineEx((Vector2){r.x+cut,r.y},(Vector2){r.x+r.width-cut,r.y},1.5f,border);
    DrawLineEx((Vector2){r.x+cut,r.y+r.height},(Vector2){r.x+r.width-cut,r.y+r.height},1.5f,border);
    DrawLineEx((Vector2){r.x,r.y+cut},(Vector2){r.x,r.y+r.height-cut},1.5f,border);
    DrawLineEx((Vector2){r.x+r.width,r.y+cut},(Vector2){r.x+r.width,r.y+r.height-cut},1.5f,border);
    DrawLineEx((Vector2){r.x,r.y+cut},(Vector2){r.x+cut,r.y},1.5f,border);
    DrawLineEx((Vector2){r.x+r.width-cut,r.y},(Vector2){r.x+r.width,r.y+cut},1.5f,border);
    DrawLineEx((Vector2){r.x,r.y+r.height-cut},(Vector2){r.x+cut,r.y+r.height},1.5f,border);
    DrawLineEx((Vector2){r.x+r.width-cut,r.y+r.height},(Vector2){r.x+r.width,r.y+r.height-cut},1.5f,border);
}

static bool CyberButton(Rectangle r, const char *label, Color accent) {
    Vector2 mp = GetMousePosition();
    bool hov = CheckCollisionPointRec(mp, r);
    bool dn  = hov && IsMouseButtonDown(MOUSE_BUTTON_LEFT);

    if (hov) {
        float p = (sinf(anim_t*6.0f)+1.0f)*0.5f;
        DrawRectangleRec((Rectangle){r.x-4,r.y-4,r.width+8,r.height+8},
            (Color){accent.r,accent.g,accent.b,(unsigned char)(20+p*25)});
    }

    Color bg = dn  ? (Color){accent.r/5,accent.g/5,accent.b/5,255}
             : hov ? (Color){accent.r/8,accent.g/8,accent.b/8,255}
             :       CB_PANEL;

    int cut = 7;
    CyberRect(r, accent, bg, cut);
    DrawRectangle(r.x+cut+1, r.y+2, r.width-cut*2-2, 2, (Color){255,255,255, hov?30:12});

    int fs = 13;
    int tw = MeasureText(label, fs);
    Color tc = hov ? accent : CB_TEXT;
    DrawText(label, (int)(r.x+(r.width-tw)/2), (int)(r.y+(r.height-fs)/2), fs, tc);

    return hov && IsMouseButtonPressed(MOUSE_BUTTON_LEFT);
}

static void CyberTextBox(Rectangle r, char *text, int *len, bool active, const char *ph) {
    Color bc = active ? CB_CYAN : CB_BLUE2;
    Color bg = active ? (Color){0x02,0x08,0x14,255} : (Color){0x03,0x06,0x10,255};

    if (active) {
        float p = (sinf(anim_t*4.0f)+1.0f)*0.5f;
        DrawRectangleRec((Rectangle){r.x-3,r.y-3,r.width+6,r.height+6},
            (Color){CB_CYAN.r,CB_CYAN.g,CB_CYAN.b,(unsigned char)(15+p*20)});
    }
    CyberRect(r, bc, bg, 6);

    if (*len == 0 && !active) {
        DrawText(ph, r.x+14, r.y+(r.height-12)/2, 12, CB_MUTED);
    } else {
        DrawText(text, r.x+14, r.y+(r.height-12)/2, 12, CB_TEXT);
        if (active && (int)(GetTime()*2)%2==0) {
            int tw = MeasureText(text, 12);
            DrawRectangle(r.x+14+tw+2, r.y+8, 2, r.height-16, CB_CYAN);
        }
    }
}

static void DrawCyberGrid(int ox, int ow, int oh) {
    int step = 44;
    float pulse = (sinf(anim_t*0.7f)+1.0f)*0.5f;
    unsigned char ga = (unsigned char)(14 + pulse*10);
    for (int x = ox; x < ox+ow; x += step)
        DrawLine(x, 0, x, oh, (Color){0x06,0x18,0x38,ga});
    for (int y = 0; y < oh; y += step)
        DrawLine(ox, y, ox+ow, y, (Color){0x06,0x18,0x38,ga});
}

static void DrawScanlines(int x, int w, int h) {
    for (int y = 0; y < h; y += 3)
        DrawRectangle(x, y, w, 1, (Color){0x00,0x00,0x00,18});
}

static void DrawTabGlow(int x, int y, int h, Color c) {
    float p = (sinf(anim_t*2.5f)+1.0f)*0.5f;
    DrawRectangle(x, y, 8, h, (Color){c.r,c.g,c.b,30});
    DrawRectangle(x, y, 3, h, (Color){c.r,c.g,c.b,(unsigned char)(140+p*115)});
}

static void DrawHexBg(int cx, int cy, float r, Color c) {
    for (int i = 0; i < 6; i++) {
        float a0 = (i    )*3.14159f/3.0f - 3.14159f/6.0f;
        float a1 = (i+1.0f)*3.14159f/3.0f - 3.14159f/6.0f;
        DrawLine(cx+(int)(cosf(a0)*r), cy+(int)(sinf(a0)*r),
                 cx+(int)(cosf(a1)*r), cy+(int)(sinf(a1)*r), c);
    }
}

static void DrawOrbitParticle(int cx, int cy, float orbit, float speed, float offset, Color c) {
    float a = anim_t * speed + offset;
    int px = cx + (int)(cosf(a)*orbit);
    int py = cy + (int)(sinf(a)*orbit);
    float p = (sinf(anim_t*4.0f+offset)+1.0f)*0.5f;
    DrawCircle(px, py, 2+(int)(p*2), (Color){c.r,c.g,c.b,(unsigned char)(120+p*135)});
}

/* =========================================================================
   MAIN LOOP
   ========================================================================= */
void iniciar_interfaz_grafica(void) {
    InitWindow(W, H, "SINA-VISUAL");
    SetTargetFPS(60);

    /* Estado inicial */
    cargar_log();
    if (mse_count > 0) red_ok = true;

    /* Intentar autolanzar carga para levantar pesos guardados si existen */
    cmd("cargar");

    char inp[100] = "";
    int  inp_len  = 0;
    bool inp_active = false;

    while (!WindowShouldClose()) {

        /* ── UPDATE ── */
        anim_t += GetFrameTime();

        /* Entrenamiento Batch (Dataset) */
        if (training && cfg_ok && ds_ok) {
            int epf = 10;
            if (epf > epoch_total - epoch_cur) epf = epoch_total - epoch_cur;
            make_dir("reports");
            FILE *lf = fopen("reports/training_log.txt", epoch_cur==0?"w":"a");
            double *exp = (double*)malloc(sizeof(double));
            for (int e = 0; e < epf; e++) {
                epoch_cur++;
                double mse = 0.0;
                for (int s = 0; s < ds->num_samples; s++) {
                    forward_propagation(red, ds->inputs[s]);
                    int ol = red->num_capas-1;
                    double out = red->capas[ol].neuronas[0].salida;
                    double tgt = ds->targets[s];
                    double df  = tgt-out; mse += df*df;
                    exp[0] = tgt; backpropagation(red, exp);
                }
                mse /= ds->num_samples;
                err_fin = mse_cur = mse;
                
                /* Visualización en consola paso a paso */
                if (epoch_cur == 1 || epoch_cur % 10 == 0 || epoch_cur == epoch_total) {
                    printf("[Entrenamiento] Epoca: %4d | Error (MSE): %.6f\n", epoch_cur, mse);
                }
                
                if (lf) fprintf(lf, "%d,%.6f\n", epoch_cur, mse);
                if (mse_count >= mse_cap) {
                    mse_cap = mse_cap ? mse_cap*2 : 512;
                    mse_hist = (float*)realloc(mse_hist, sizeof(float)*mse_cap);
                }
                mse_hist[mse_count++] = (float)mse;
            }
            if (lf) fclose(lf);
            free(exp);
            if (epoch_cur >= epoch_total) {
                training = false; red_ok = true;
                guardar_pesos(red, "data/red_memoria.dat"); /* GUARDAR AL FINALIZAR */
                
                /* Graficar el error en ASCII directamente en la terminal */
                printf("\n=== GRAFICO DE ERROR ASCII (Loss) ===\n");
                float max_mse = 0;
                for(int i=0; i<mse_count; i++) if(mse_hist[i] > max_mse) max_mse = mse_hist[i];
                if(max_mse == 0) max_mse = 1;
                
                int graph_h = 10;
                for (int y = graph_h; y >= 0; y--) {
                    printf("%5.4f | ", (float)y/graph_h * max_mse);
                    for (int x = 0; x < 40; x++) {
                        int idx = (int)((float)x / 39 * (mse_count - 1));
                        float val = mse_hist[idx];
                        int bar_h = (int)((val / max_mse) * graph_h);
                        if (bar_h == y) printf("*");
                        else if (bar_h > y) printf("#");
                        else printf(" ");
                    }
                    printf("\n");
                }
                printf("        +----------------------------------------\n");
                printf("          Progreso (Epocas)\n=====================================\n");
                
                char buf[128];
                snprintf(buf, sizeof(buf),
                    "[OK] Entrenamiento completo y guardado. MSE final: %.6f", err_fin);
                add_msg(buf, 0);
            }
        }

        /* Drag & Drop de Radiografías */
        if (IsFileDropped()) {
            FilePathList droppedFiles = LoadDroppedFiles();
            if (droppedFiles.count > 0 && tab == 3 && cfg_ok && red_ok) {
                if (rx_loaded) {
                    UnloadTexture(current_rx_texture);
                    UnloadImage(current_rx_img);
                }
                current_rx_img = LoadImage(droppedFiles.paths[0]);
                if (current_rx_img.width > 0) {
                    /* Redimensionar la radiografía completa a 64x64 */

                    /* Redimensionar a 64x64 para 4096 entradas */
                    ImageResize(&current_rx_img, 64, 64);
                    ImageColorGrayscale(&current_rx_img);
                    Color *pixels = LoadImageColors(current_rx_img);
                    for (int i = 0; i < 4096; i++) {
                        rx_features[i] = (double)pixels[i].r / 255.0; 
                    }
                    UnloadImageColors(pixels);

                    /* Imagen para visualización */
                    Image display_img = LoadImage(droppedFiles.paths[0]);
                    ImageResize(&display_img, 200, 200);
                    current_rx_texture = LoadTextureFromImage(display_img);
                    UnloadImage(display_img);

                    /* Rendering in Console (ASCII Graphic) as requested by Rubric */
                printf("\n--- IMAGEN RADIOGRAFICA (64x64 ASCII) ---\n");
                const char *ascii_chars = " .:-=+*#%@";
                for (int r = 0; r < 64; r+=2) {
                    for (int c = 0; c < 64; c++) {
                        double val = rx_features[r*64+c];
                        int idx = (int)(val * 9.0);
                        if (idx < 0) idx = 0;
                        if (idx > 9) idx = 9;
                        printf("%c", ascii_chars[idx]);
                    }
                    printf("\n");
                }
                printf("-----------------------------------------\n");

                rx_loaded = true;
                    pred_done = false;
                    add_msg("[RX] Radiografia cargada correctamente. Lista para analisis.", 0);
                }
            } else {
                if (tab != 3) add_msg("[!] Cambia a la pestaña Analizador RX para cargar imagenes.", 2);
                else add_msg("[!] Carga y prepara la red primero (Comando: cargar).", 2);
            }
            UnloadDroppedFiles(droppedFiles);
        }

        /* Input teclado */
        if (inp_active) {
            int k = GetCharPressed();
            while (k > 0) {
                if (k >= 32 && k <= 125 && inp_len < 99)
                    { inp[inp_len++] = (char)k; inp[inp_len] = '\0'; }
                k = GetCharPressed();
            }
            if (IsKeyPressed(KEY_BACKSPACE) && inp_len > 0) inp[--inp_len] = '\0';
            if (IsKeyPressed(KEY_ENTER) && inp_len > 0)
                { cmd(inp); inp[0] = '\0'; inp_len = 0; }
        }
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            Rectangle ibx = {SIDEBAR, H-80, W-SIDEBAR, 38};
            inp_active = CheckCollisionPointRec(GetMousePosition(), ibx);
        }

        /* ── DRAW ── */
        BeginDrawing();
        ClearBackground(CB_BG);

        /* === FONDO GLOBAL === */
        DrawCyberGrid(SIDEBAR, W-SIDEBAR, H);
        float hpulse = (sinf(anim_t*0.4f)+1.0f)*0.5f;
        DrawHexBg(950, 120, 90+(int)(hpulse*8), (Color){0x00,0xFF,0xF5,8});
        DrawHexBg(950, 120, 55+(int)(hpulse*5), (Color){0x00,0xFF,0xF5,12});
        DrawHexBg(320, 560, 60+(int)(hpulse*6), (Color){0x20,0x80,0xFF,10});
        DrawHexBg(680, 360, 120+(int)(hpulse*10),(Color){0xB0,0x00,0xFF,6});

        DrawOrbitParticle(950, 120, 110, 0.3f, 0.0f,   CB_CYAN);
        DrawOrbitParticle(950, 120, 110, 0.3f, 2.094f, CB_PURPLE);
        DrawOrbitParticle(950, 120, 110, 0.3f, 4.188f, CB_BLUE);
        DrawOrbitParticle(320, 560, 75,  0.5f, 1.0f,   CB_CYAN);
        DrawOrbitParticle(680, 360, 140, 0.2f, 3.0f,   CB_PURPLE);

        float pa = (sinf(anim_t*0.45f)+1.0f)*0.5f;
        DrawCircle(940, 110, 130, (Color){0x00,0xFF,0xF5,(unsigned char)(4+pa*5)});
        DrawCircle(310, 570, 90,  (Color){0x20,0x80,0xFF,(unsigned char)(4+pa*4)});
        DrawCircle(680, 360, 170, (Color){0xB0,0x00,0xFF,(unsigned char)(3+pa*4)});
        DrawScanlines(SIDEBAR, W-SIDEBAR, H);

        /* =====================================================================
           SIDEBAR
           ===================================================================== */
        DrawRectangle(0, 0, SIDEBAR-2, H, CB_SIDE);
        float bpa = (sinf(anim_t*1.8f)+1.0f)*0.5f;
        DrawRectangle(SIDEBAR-4, 0, 4, H, (Color){CB_CYAN.r,CB_CYAN.g,CB_CYAN.b,(unsigned char)(40+bpa*80)});
        DrawRectangle(SIDEBAR-2, 0, 2, H, (Color){CB_CYAN.r,CB_CYAN.g,CB_CYAN.b,(unsigned char)(100+bpa*155)});
        DrawRectangle(0, 0, SIDEBAR-2, 3, CB_CYAN);
        DrawScanlines(0, SIDEBAR-2, H);

        /* ── LOGO ── */
        {
            Rectangle lr = {8, 8, SIDEBAR-18, 68};
            CyberRect(lr, CB_CYAN, (Color){0x03,0x07,0x12,255}, 10);
            DrawRectangle(8+11, 8, 60, 3, CB_CYAN);

            int bx=24, by=28;
            float ap2 = (sinf(anim_t*2.0f)+1.0f)*0.5f;
            DrawCircle(bx, by, 5, CB_BLUE); DrawCircle(bx, by+16, 5, CB_BLUE);
            DrawCircle(bx+20, by+8, 5, CB_CYAN); DrawCircle(bx+40, by+8, 5, CB_PURPLE);
            DrawLine(bx,bx+by-bx,bx+20,by+8, (Color){CB_CYAN.r,CB_CYAN.g,CB_CYAN.b,(unsigned char)(100+ap2*100)});
            DrawLine(bx, by+16, bx+20, by+8, (Color){CB_CYAN.r,CB_CYAN.g,CB_CYAN.b,(unsigned char)(100+ap2*100)});
            DrawLine(bx+20, by+8, bx+40, by+8, (Color){CB_PURPLE.r,CB_PURPLE.g,CB_PURPLE.b,(unsigned char)(100+ap2*100)});
            DrawCircleLines(bx+20, by+8, 8+ap2*2, (Color){CB_CYAN.r,CB_CYAN.g,CB_CYAN.b,(unsigned char)(40+ap2*30)});
            DrawCircleLines(bx+40, by+8, 8+ap2*2, (Color){CB_PURPLE.r,CB_PURPLE.g,CB_PURPLE.b,(unsigned char)(40+ap2*30)});

            DrawText("SINA-VISUAL",  bx+50, 22, 15, CB_TEXT);
            DrawText("// DENTAL ANALYZER", bx+50, 40, 9, CB_MUTED);
            DrawText("v4.0 AI",      bx+50, 54, 9, CB_CYAN);
        }

        /* ── TABS ── */
        const char *tab_names[] = {"Consola IA","Red Neuronal","Entrenamiento","Analizador RX"};
        const char *tab_pfx[]   = {"//","[]","~~","**"};
        Color tab_colors[]      = {CB_CYAN, CB_BLUE, CB_PURPLE, CB_GREEN};

        for (int i = 0; i < 4; i++) {
            int ty = 88 + i*62;
            Rectangle tr = {8, ty, SIDEBAR-18, 52};
            Vector2 mp2  = GetMousePosition();
            bool hov = CheckCollisionPointRec(mp2, tr);
            bool act = (tab == i);

            if (act) {
                DrawRectangle(8, ty, SIDEBAR-18, 52, (Color){tab_colors[i].r/8, tab_colors[i].g/8, tab_colors[i].b/8, 220});
                GlowLine(8, ty, SIDEBAR-10, ty, tab_colors[i], 1.0f);
                GlowLine(8, ty+52, SIDEBAR-10, ty+52, (Color){tab_colors[i].r,tab_colors[i].g,tab_colors[i].b,60}, 0.5f);
                DrawTabGlow(8, ty+6, 40, tab_colors[i]);
            } else if (hov) {
                DrawRectangle(8, ty, SIDEBAR-18, 52, (Color){0x05,0x10,0x22,180});
                DrawLine(8, ty, SIDEBAR-10, ty, (Color){CB_BLUE2.r,CB_BLUE2.g,CB_BLUE2.b,80});
            }

            Color tc = act ? tab_colors[i] : (hov ? CB_TEXT : CB_MUTED);
            DrawText(tab_pfx[i], 22, ty+19, 11, act ? tab_colors[i] : CB_MUTED2);
            DrawText(tab_names[i], 48, ty+18, 14, tc);

            if (hov && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) tab = i;
        }

        DrawRectangle(12, 88+4*62+4, SIDEBAR-24, 1, CB_MUTED2);

        /* ── Chip de modo ── */
        {
            Color chc = training ? CB_AMBER : (red_ok ? CB_GREEN : CB_BLUE);
            const char *chl = training ? "ENTRENANDO" : (red_ok ? "RED LISTA" : "STANDBY");
            float cp = (sinf(anim_t*3.5f)+1.0f)*0.5f;

            Rectangle cr = {12, H-70, SIDEBAR-26, 34};
            CyberRect(cr, chc, (Color){0x03,0x07,0x10,255}, 5);
            DrawCircle(28, H-53, 5, (Color){chc.r,chc.g,chc.b,(unsigned char)(30+cp*40)});
            DrawCircle(28, H-53, 3, (Color){chc.r,chc.g,chc.b,(unsigned char)(180+cp*75)});
            DrawText(chl, 40, H-60, 11, chc);

            for (int i = 0; i < 3; i++) {
                float llen = 30.0f + sinf(anim_t*2.0f+i*1.5f)*20.0f;
                DrawRectangle(12+i*40, H-32, (int)llen, 2, (Color){CB_MUTED2.r,CB_MUTED2.g,CB_MUTED2.b,100});
            }
        }

        /* =====================================================================
           HEADER CONTENIDO
           ===================================================================== */
        DrawRectangle(SIDEBAR, 0, W-SIDEBAR, 74, CB_PANEL);
        GlowLine(SIDEBAR, 74, W, 74, CB_CYAN, 0.5f);
        DrawRectangle(SIDEBAR, 0, W-SIDEBAR, 2, CB_CYAN);

        struct { const char *a; const char *b; Color c; } hdrs[] = {
            {"CONSOLA"," IA",      CB_CYAN},
            {"RED",    " NEURONAL",CB_BLUE},
            {"ENTRE",  "NAMIENTO", CB_PURPLE},
            {"ANALIZA","DOR RX",   CB_GREEN},
        };
        const char *subs[] = {
            "Interfaz de comandos y chat con la red",
            "Visualizacion de capas y pesos sinapticos",
            "Historico del error de entrenamiento (MSE)",
            "Carga de radiografías, analisis y aprendizaje continuo",
        };

        int ha = MeasureText(hdrs[tab].a, 28);
        DrawText(hdrs[tab].a, SIDEBAR+15, 10, 28, CB_TEXT);
        DrawText(hdrs[tab].b, SIDEBAR+15+ha, 10, 28, hdrs[tab].c);
        DrawText(subs[tab], SIDEBAR+15, 48, 11, CB_MUTED);

        int slen = MeasureText(subs[tab], 11);
        GlowLine(SIDEBAR+15, 44, SIDEBAR+15+slen, 44, (Color){hdrs[tab].c.r,hdrs[tab].c.g,hdrs[tab].c.b,60}, 0.5f);

        {
            Color chc = training ? CB_AMBER : (red_ok ? CB_GREEN : CB_BLUE);
            const char *chl = training ? "ENTRENANDO" : (red_ok ? "LISTO" : "STANDBY");
            int cw2 = MeasureText(chl, 11) + 38;
            int cx2 = W - cw2 - 14;
            CyberRect((Rectangle){cx2,18,cw2,34}, chc, CB_PANEL, 5);
            float cp = (sinf(anim_t*3.5f)+1.0f)*0.5f;
            DrawCircle(cx2+12, 35, 4, (Color){chc.r,chc.g,chc.b,(unsigned char)(160+cp*75)});
            DrawText(chl, cx2+22, 28, 11, chc);
        }

        /* =====================================================================
           TAB 0 — CONSOLA IA
           ===================================================================== */
        if (tab == 0) {
            DrawRectangle(SIDEBAR, 74, W-SIDEBAR, H-74-84, (Color){0x03,0x05,0x0E,255});
            DrawScanlines(SIDEBAR, W-SIDEBAR, H-74-84+74);

            int dy = 74 + (H-74-84) - 8;
            for (int i = chat_count-1; i >= 0; i--) {
                const char *t = chat_msg[i];
                int mt = chat_type[i];

                int nl = 1;
                for (int c2 = 0; t[c2]; c2++) if (t[c2]=='\n') nl++;
                int mh = nl*17 + 16;
                dy -= mh;
                if (dy < 80) break;

                int mxw = 0;
                char lb[256]; int li = 0;
                for (int c2 = 0;; c2++) {
                    if (t[c2]=='\n'||t[c2]=='\0') {
                        lb[li] = '\0';
                        int lw = MeasureText(lb, 12); if (lw > mxw) mxw = lw;
                        li = 0; if (t[c2]=='\0') break;
                    } else if (li < 255) lb[li++] = t[c2];
                }
                int bw = mxw + 24; if (bw < 90) bw = 90;
                int bx = SIDEBAR + 14;
                if (mt == 1) bx = W - 14 - bw;

                Color bg2 = (mt==1) ? (Color){0x06,0x12,0x24,220} :(mt==2) ? (Color){0x18,0x04,0x06,220} : (Color){0x04,0x09,0x16,220};
                Color bc2 = (mt==1) ? CB_BLUE :(mt==2) ? CB_RED : CB_CYAN;

                CyberRect((Rectangle){bx,dy,bw,mh}, bc2, bg2, 6);

                int ty2 = dy+8; li = 0;
                for (int c2 = 0;; c2++) {
                    if (t[c2]=='\n'||t[c2]=='\0') {
                        lb[li] = '\0';
                        DrawText(lb, bx+12, ty2, 12, CB_TEXT);
                        ty2 += 17; li = 0; if (t[c2]=='\0') break;
                    } else if (li < 255) lb[li++] = t[c2];
                }
                dy -= 5;
            }

            const char *btns[] = {"Cargar","Entrenar","Analizador","Reporte"};
            Color baccs[] = {CB_BLUE, training?CB_AMBER:CB_GREEN, CB_CYAN, CB_MUTED};
            int bw2 = (W - SIDEBAR) / 4;
            for (int i = 0; i < 4; i++) {
                Rectangle br = {SIDEBAR+(float)i*bw2, H-84, (float)bw2-2, 40};
                if (CyberButton(br, btns[i], baccs[i])) {
                    const char *cs[] = {"cargar","entrenar","analizador","reporte"};
                    cmd(cs[i]);
                }
            }

            CyberTextBox((Rectangle){SIDEBAR, H-42, W-SIDEBAR, 42}, inp, &inp_len, inp_active,
                "// Seleccione una opcion haciendo clic en los botones superiores");

        /* =====================================================================
           TAB 1 — RED NEURONAL
           ===================================================================== */
        } else if (tab == 1) {
            if (!cfg_ok) {
                DrawText("Carga la configuracion en Consola IA primero.", SIDEBAR+80, H/2, 15, CB_MUTED);
            } else {
                int layers = cfg->num_capas;
                int area_x = SIDEBAR+10, area_w = W-SIDEBAR-20;
                int area_y = 84, area_h = H-84-120;

                for (int i = 0; i < layers-1; i++) {
                    int nc = cfg->neuronas_por_capa[i];
                    int nn = cfg->neuronas_por_capa[i+1];
                    float x0 = area_x + ((float)(i+1)/(layers+1))*area_w;
                    float x1 = area_x + ((float)(i+2)/(layers+1))*area_w;
                    
                    /* Limitar dibujado para capas grandes (ej 100 nodos) */
                    int draw_nc = nc > 10 ? 10 : nc;
                    int draw_nn = nn > 10 ? 10 : nn;

                    for (int j = 0; j < draw_nc; j++) {
                        float y0 = area_y + ((float)(j+1)/(draw_nc+1))*area_h;
                        for (int k = 0; k < draw_nn; k++) {
                            float y1 = area_y + ((float)(k+1)/(draw_nn+1))*area_h;
                            Color lc = {CB_BLUE2.r,CB_BLUE2.g,CB_BLUE2.b,20};
                            float th = 0.8f;
                            if (red_ok && red) {
                                double w2 = red->capas[i+1].neuronas[k].pesos[j];
                                th = (float)fabs(w2)*2.5f;
                                if (th < 0.5f) th = 0.5f;
                                if (th > 4.0f) th = 4.0f;
                                unsigned char al=(unsigned char)(25+(fabs(w2)/(fabs(w2)+0.5))*130);
                                lc = w2>0 ? (Color){CB_CYAN.r,CB_CYAN.g,CB_CYAN.b,al} : (Color){CB_PURPLE.r,CB_PURPLE.g,CB_PURPLE.b,al};
                            }
                            DrawLineEx((Vector2){x0,y0},(Vector2){x1,y1}, th, lc);
                        }
                    }
                }

                Color layer_cols[] = {CB_BLUE, CB_CYAN, CB_GREEN};
                const char *layer_names[] = {"ENTRADA","OCULTA","SALIDA"};

                for (int i = 0; i < layers; i++) {
                    int n = cfg->neuronas_por_capa[i];
                    float x = area_x + ((float)(i+1)/(layers+1))*area_w;
                    int ci = (i==0)?0:(i==layers-1)?2:1;
                    Color nc2 = layer_cols[ci];

                    int draw_n = n > 10 ? 10 : n;

                    for (int j = 0; j < draw_n; j++) {
                        float y = area_y + ((float)(j+1)/(draw_n+1))*area_h;
                        float hp = (sinf(anim_t*2.0f+i*0.7f+j*0.4f)+1.0f)*0.5f;

                        DrawCircle(x, y, 22, (Color){nc2.r,nc2.g,nc2.b,(unsigned char)(8+hp*12)});
                        DrawCircleLines(x, y, 15, (Color){nc2.r,nc2.g,nc2.b,(unsigned char)(150+hp*80)});
                        DrawCircle(x, y, 12, (Color){0x02,0x05,0x0C,255});
                        DrawCircle(x, y, 8, nc2);
                        DrawCircle(x-2, y-2, 3, (Color){255,255,255,60});

                        Vector2 mp3 = GetMousePosition();
                        if (CheckCollisionPointCircle(mp3,(Vector2){x,y},15.0f)) {
                            char tt[128];
                            if (i > 0 && red_ok && red) snprintf(tt,sizeof(tt),"N%d | bias=%.4f",j+1,red->capas[i].neuronas[j].bias);
                            else snprintf(tt,sizeof(tt),"N%d",j+1);
                            int tw2 = MeasureText(tt,11);
                            CyberRect((Rectangle){mp3.x+14,mp3.y-16,tw2+20,26}, nc2,(Color){0x03,0x07,0x12,230},4);
                            DrawText(tt, mp3.x+22, mp3.y-10, 11, CB_TEXT);
                        }
                    }

                    const char *lname = layer_names[ci];
                    DrawText(lname, x-MeasureText(lname,9)/2, area_y+area_h+10, 9, nc2);
                    char nstr[16]; snprintf(nstr,sizeof(nstr),"(%d nodos)",n);
                    DrawText(nstr, x-MeasureText(nstr,9)/2, area_y+area_h+22, 9, CB_MUTED);
                }
            }

            DrawRectangle(SIDEBAR, H-116, W-SIDEBAR, 116, CB_PANEL);
            GlowLine(SIDEBAR, H-116, W, H-116, CB_BLUE, 0.5f);

            const char *lbs[] = {"CAPAS","ARQUITECTURA","LEARNING RATE","EPOCAS"};
            char vals[4][64]  = {"--","--","--","--"};
            if (cfg_ok) {
                snprintf(vals[0],64,"%d",cfg->num_capas);
                snprintf(vals[2],64,"%.4f",cfg->tasa_aprendizaje);
                snprintf(vals[3],64,"%d",cfg->epocas);
                int al = 0;
                for (int i = 0; i < cfg->num_capas; i++)
                    al += snprintf(vals[1]+al,64-al,"%d%s",
                        cfg->neuronas_por_capa[i], i<cfg->num_capas-1?"-":"");
            }
            for (int i = 0; i < 4; i++) {
                int cx3 = SIDEBAR+20+i*210;
                DrawText(lbs[i],  cx3, H-102, 10, CB_MUTED);
                DrawText(vals[i], cx3, H-82,  20, i%2?CB_CYAN:CB_BLUE);
            }

        /* =====================================================================
           TAB 2 — ENTRENAMIENTO
           ===================================================================== */
        } else if (tab == 2) {
            DrawRectangle(SIDEBAR, 74, W-SIDEBAR, 68, CB_PANEL);
            GlowLine(SIDEBAR, 142, W, 142, CB_PURPLE, 0.5f);

            char sv[4][32] = {"--","--","--","--"};
            if (mse_count > 0) {
                float ie = mse_hist[0], fe = mse_hist[mse_count-1];
                float rd = ie>0 ? ((ie-fe)/ie)*100.0f : 0;
                snprintf(sv[0],32,"%.6f",ie);
                snprintf(sv[1],32,"%.6f",fe);
                snprintf(sv[2],32,"%.1f%%",rd);
                snprintf(sv[3],32,"%d",epoch_cur);
            }
            const char *sl[] = {"ERROR INICIAL","ERROR FINAL","REDUCCION","EPOCAS"};
            Color sc[] = {CB_MUTED, CB_GREEN, CB_CYAN, CB_BLUE};
            for (int i = 0; i < 4; i++) {
                int sx = SIDEBAR+20+i*210;
                DrawText(sl[i], sx, 84,  10, CB_MUTED);
                DrawText(sv[i], sx, 100, 16, sc[i]);
            }

            if (!mse_count) {
                DrawText("Entrena la red en Consola IA para ver el grafico.", SIDEBAR+100, H/2, 14, CB_MUTED);
            } else {
                int gx = SIDEBAR+60, gy = 154, gw = W-SIDEBAR-80, gh = H-154-28;
                DrawRectangle(gx, gy, gw, gh, (Color){0x02,0x04,0x0C,255});
                GlowLine(gx, gy,    gx+gw, gy,    (Color){CB_PURPLE.r,CB_PURPLE.g,CB_PURPLE.b,50}, 0.5f);
                DrawLine(gx, gy+gh, gx+gw, gy+gh, CB_BLUE2);
                DrawLine(gx, gy,    gx,    gy+gh,  CB_BLUE2);

                for (int i = 1; i < 5; i++) {
                    int yg = gy + i*gh/5;
                    DrawLine(gx, yg, gx+gw, yg, (Color){0x08,0x18,0x30,60});
                    int xg = gx + i*gw/5;
                    DrawLine(xg, gy, xg, gy+gh, (Color){0x08,0x18,0x30,40});
                }

                float mx = 0;
                for (int i = 0; i < mse_count; i++) if (mse_hist[i] > mx) mx = mse_hist[i];
                if (mx == 0) mx = 1;

                for (int i = 0; i < mse_count-1; i++) {
                    float x0 = gx+(float)i/(mse_count-1)*gw;
                    float x1 = gx+(float)(i+1)/(mse_count-1)*gw;
                    float y0 = gy+gh-(mse_hist[i]/mx)*gh;
                    float y1 = gy+gh-(mse_hist[i+1]/mx)*gh;
                    DrawTriangle((Vector2){x0,y0},(Vector2){x0,(float)(gy+gh)},(Vector2){x1,(float)(gy+gh)}, (Color){CB_PURPLE.r,CB_PURPLE.g,CB_PURPLE.b,10});
                    DrawTriangle((Vector2){x0,y0},(Vector2){x1,(float)(gy+gh)},(Vector2){x1,y1}, (Color){CB_PURPLE.r,CB_PURPLE.g,CB_PURPLE.b,10});
                    DrawLineEx((Vector2){x0,y0},(Vector2){x1,y1},3.0f, (Color){CB_PURPLE.r,CB_PURPLE.g,CB_PURPLE.b,60});
                    DrawLineEx((Vector2){x0,y0},(Vector2){x1,y1},1.5f, CB_PURPLE);
                }

                if (mse_count > 1) {
                    float lx = gx+gw;
                    float ly = gy+gh-(mse_hist[mse_count-1]/mx)*gh;
                    float pr = (sinf(anim_t*5.0f)+1.0f)*0.5f;
                    DrawCircle(lx, ly, 10+pr*4, (Color){CB_CYAN.r,CB_CYAN.g,CB_CYAN.b,(unsigned char)(30+pr*40)});
                    DrawCircle(lx, ly, 5, CB_CYAN);
                }

                for (int i = 0; i <= 4; i++) {
                    float yg  = gy + i*gh/5;
                    float val2 = (1.0f - i/4.0f)*mx;
                    char lb2[16]; snprintf(lb2,sizeof(lb2),"%.4f",val2);
                    DrawText(lb2, gx-58, yg-6, 9, CB_MUTED);
                }

                if (training && epoch_total > 0) {
                    float pct = (float)epoch_cur/epoch_total;
                    DrawRectangle(gx, gy+gh+8, gw, 6, (Color){0x06,0x10,0x24,255});
                    DrawRectangle(gx, gy+gh+8, (int)(pct*gw), 6, CB_PURPLE);
                    DrawRectangle(gx+(int)(pct*gw)-4, gy+gh+6, 8, 10, CB_CYAN);
                    char pb[32];
                    snprintf(pb,sizeof(pb),"%.0f%%  Ep.%d/%d",pct*100,epoch_cur,epoch_total);
                    DrawText(pb, gx+gw/2-MeasureText(pb,10)/2, gy+gh+20, 10, CB_PURPLE);
                }
            }

        /* =====================================================================
           TAB 3 — ANALIZADOR RX (DRAG & DROP)
           ===================================================================== */
        } else if (tab == 3) {
            if (!cfg_ok || !red_ok) {
                DrawText("La red no esta inicializada. Ejecuta 'cargar' en Consola.", SIDEBAR+100, H/2, 14, CB_MUTED);
            } else if (cfg->neuronas_por_capa[0] != 4096) {
                DrawText("ERROR: La capa de entrada debe ser 4096 para la matriz 64x64.", SIDEBAR+100, H/2, 14, CB_RED);
            } else {
                
                int cx3 = W - 230, cy3 = H/2 - 20;

                if (!rx_loaded) {
                    /* Zona de Drop */
                    Rectangle dropRect = {SIDEBAR+40, 120, W-SIDEBAR-350, 340};
                    float pulse = (sinf(anim_t*2.0f)+1.0f)*0.5f;
                    DrawRectangleRec(dropRect, (Color){0x04,0x08,0x14,180});
                    DrawRectangleLinesEx(dropRect, 2.0f, (Color){CB_CYAN.r,CB_CYAN.g,CB_CYAN.b,(unsigned char)(50+pulse*100)});
                    
                    const char *txt1 = "Arrastra una Radiografia aqui";
                    const char *txt2 = "(PNG/JPG)";
                    DrawText(txt1, dropRect.x + dropRect.width/2 - MeasureText(txt1, 20)/2, dropRect.y + dropRect.height/2 - 30, 20, CB_TEXT);
                    DrawText(txt2, dropRect.x + dropRect.width/2 - MeasureText(txt2, 14)/2, dropRect.y + dropRect.height/2,   14, CB_MUTED);
                    const char *txt3 = "- o -";
                    DrawText(txt3, dropRect.x + dropRect.width/2 - MeasureText(txt3, 13)/2, dropRect.y + dropRect.height/2 + 24, 13, CB_MUTED);

                    /* Boton explorador de archivos */
                    Rectangle btnBrowse = {dropRect.x + dropRect.width/2 - 110, dropRect.y + dropRect.height/2 + 46, 220, 40};
                    if (CyberButton(btnBrowse, "[ SELECCIONAR ARCHIVO ]", CB_PURPLE)) {
                        char picked[1024] = {0};
                        if (open_file_dialog(picked, sizeof(picked))) {
                            if (current_rx_img.data) UnloadImage(current_rx_img);
                            if (current_rx_texture.id) UnloadTexture(current_rx_texture);
                            current_rx_img = LoadImage(picked);
                            if (current_rx_img.width > 0) {
                                /* Redimensionar la radiografía completa a 64x64 */
                                ImageResize(&current_rx_img, 64, 64);
                                ImageColorGrayscale(&current_rx_img);
                                /* Extraer features directamente */
                                Color *px = LoadImageColors(current_rx_img);
                                if (px) {
                                    for (int pi = 0; pi < 4096; pi++)
                                        rx_features[pi] = px[pi].r / 255.0;
                                    UnloadImageColors(px);
                                }
                                /* Escalar para visualizacion */
                                Image disp = ImageCopy(current_rx_img);
                                ImageResize(&disp, 200, 200);
                                current_rx_texture = LoadTextureFromImage(disp);
                                UnloadImage(disp);
                                rx_loaded  = true;
                                pred_done  = false;
                            }
                        }
                    }
                } else {
                    /* Visor de la radiografía cargada */
                    Rectangle viewRect = {SIDEBAR+40, 120, 220, 220};
                    CyberRect(viewRect, CB_CYAN, CB_BG, 4);
                    DrawTexture(current_rx_texture, viewRect.x+10, viewRect.y+10, WHITE);
                    DrawText("IMAGEN CARGADA", viewRect.x+10, viewRect.y+235, 10, CB_MUTED);
                    /* Botones de accion */
                    bool do_pred = CyberButton((Rectangle){SIDEBAR+40, 355, 220, 40}, "[ ANALIZAR RX ]", CB_CYAN);
                    if (do_pred && red_ok) {
                        forward_propagation(red, rx_features);
                        int ol = red->num_capas-1;
                        pred_out  = red->capas[ol].neuronas[0].salida;
                        pred_done = true;
                        printf("\n[Inferencia] Diagnostico emitido: %.4f (%s)\n",
                               pred_out, pred_out >= 0.5 ? "TEJIDO SANO" : "CARIES");
                    }
                    /* Boton cargar otra imagen */
                    if (CyberButton((Rectangle){SIDEBAR+40, 403, 220, 34}, "[ CARGAR OTRA ]", CB_MUTED2)) {
                        char picked[1024] = {0};
                        if (open_file_dialog(picked, sizeof(picked))) {
                            if (current_rx_img.data) UnloadImage(current_rx_img);
                            if (current_rx_texture.id) UnloadTexture(current_rx_texture);
                            current_rx_img = LoadImage(picked);
                            if (current_rx_img.width > 0) {
                                ImageResize(&current_rx_img, 64, 64);
                                ImageColorGrayscale(&current_rx_img);
                                Color *px = LoadImageColors(current_rx_img);
                                if (px) {
                                    for (int pi = 0; pi < 4096; pi++)
                                        rx_features[pi] = px[pi].r / 255.0;
                                    UnloadImageColors(px);
                                }
                                Image disp = ImageCopy(current_rx_img);
                                ImageResize(&disp, 200, 200);
                                current_rx_texture = LoadTextureFromImage(disp);
                                UnloadImage(disp);
                                pred_done = false;
                            } else {
                                rx_loaded = false;
                            }
                        }
                    }

                    /* Mini visualización de matriz 64x64 normalizada */
                    int mx = SIDEBAR+280, my = 120, mcell = 2;
                    DrawText("Matriz de Densidades (Input)", mx, my-18, 10, CB_MUTED);
                    for (int r = 0; r < 64; r++) {
                        for (int c = 0; c < 64; c++) {
                            double val = rx_features[r*64+c];
                            unsigned char g = (unsigned char)(val * 255.0);
                            DrawRectangle(mx+c*mcell, my+r*mcell, mcell, mcell, (Color){g,g,g,255});
                        }
                    }
                    DrawRectangleLines(mx-2, my-2, 64*mcell+3, 64*mcell+3, CB_MUTED2);
                }

                /* Dial de Resultado (derecha) */
                if (!pred_done) {
                    DrawText("DIAGNOSTICO", cx3-MeasureText("DIAGNOSTICO",10)/2, cy3-80, 10, CB_MUTED);
                    float pp = (sinf(anim_t*1.2f)+1.0f)*0.5f;
                    DrawCircleLines(cx3, cy3, 72, (Color){CB_MUTED2.r,CB_MUTED2.g,CB_MUTED2.b,100});
                    DrawCircleLines(cx3, cy3, 65, (Color){CB_MUTED2.r,CB_MUTED2.g,CB_MUTED2.b,(unsigned char)(40+pp*40)});
                    DrawText("--", cx3-MeasureText("--",20)/2, cy3-10, 20, CB_MUTED2);
                } else {
                    /* Lógica: > 0.5 = Sano (Claro), < 0.5 = Caries (Oscuro) */
                    bool sano = pred_out >= 0.5;
                    Color rc = sano ? CB_GREEN : CB_RED;
                    float rp = (sinf(anim_t*2.5f)+1.0f)*0.5f;

                    DrawText("DIAGNOSTICO", cx3-MeasureText("DIAGNOSTICO",10)/2, cy3-90, 10, CB_MUTED);

                    DrawCircle(cx3, cy3, 80+rp*5, (Color){rc.r,rc.g,rc.b,(unsigned char)(8+rp*10)});
                    DrawCircle(cx3, cy3, 72, (Color){rc.r,rc.g,rc.b,30});
                    DrawHexBg(cx3, cy3, 90+(int)(rp*6), (Color){rc.r,rc.g,rc.b,(unsigned char)(20+rp*15)});
                    DrawCircleSector((Vector2){(float)cx3,(float)cy3}, 68.0f, -90.0f, -90.0f+(float)(pred_out*360), 64, (Color){rc.r,rc.g,rc.b,160});
                    DrawCircle(cx3, cy3, 54, (Color){0x02,0x04,0x0A,255});
                    DrawCircleLines(cx3, cy3, 68, rc);

                    char os[16]; snprintf(os, sizeof(os), "%.4f", pred_out);
                    DrawText(os, cx3-MeasureText(os,20)/2, cy3-13, 20, rc);

                    const char *vd = sano ? "TEJIDO SANO" : "CARIES DETECTADA";
                    DrawText(vd, cx3-MeasureText(vd,12)/2, cy3+12, 12, rc);
                    DrawText("umbral 0.5", cx3-MeasureText("umbral 0.5",9)/2, cy3+30, 9, CB_MUTED);

                    int bw3 = MeasureText(vd,14)+30;
                    CyberRect((Rectangle){cx3-bw3/2,cy3+60,bw3,32}, rc, (Color){rc.r/6,rc.g/6,rc.b/6,240}, 5);
                    DrawText(vd, cx3-MeasureText(vd,14)/2, cy3+67, 14, rc);

                    /* CONTROLES DE FEEDBACK (Aprendizaje Continuo) */
                    DrawText("FEEDBACK DE EXPERTO (APRENDIZAJE CONTINUO)", cx3-MeasureText("FEEDBACK DE EXPERTO (APRENDIZAJE CONTINUO)",10)/2, cy3+120, 10, CB_CYAN);
                    
                    bool f_sano = CyberButton((Rectangle){cx3-95, cy3+140, 90, 36}, "ES SANO", CB_GREEN);
                    bool f_cari = CyberButton((Rectangle){cx3+5,  cy3+140, 90, 36}, "ES CARIES", CB_RED);

                    if (f_sano || f_cari) {
                        double expected = f_sano ? 1.0 : 0.0;
                        
                        /* 1. Guardar primero en el dataset para incluir la nueva muestra */
                        if (cfg_ok && cfg->dataset_path[0] != '\0') {
                            FILE *f = fopen(cfg->dataset_path, "a");
                            if (f) {
                                for (int i = 0; i < 4096; i++) {
                                    fprintf(f, "%.4f,", rx_features[i]);
                                }
                                fprintf(f, "%.1f\n", expected);
                                fclose(f);
                                
                                /* Recargar dataset para incluir la nueva fila */
                                if (ds) {
                                    liberar_dataset(ds);
                                }
                                ds = cargar_dataset(cfg->dataset_path);
                                if (ds) ds_ok = true;
                            }
                        }

                        /* 2. Re-entrenar con TODO el dataset para evitar el olvido catastrófico */
                        if (ds_ok && ds->num_samples > 0) {
                            double *exp = (double*)malloc(sizeof(double));
                            for(int ep = 0; ep < 30; ep++) { /* 30 épocas de ajuste fino */
                                for (int s = 0; s < ds->num_samples; s++) {
                                    forward_propagation(red, ds->inputs[s]);
                                    exp[0] = ds->targets[s];
                                    backpropagation(red, exp);
                                }
                            }
                            free(exp);
                        } else {
                            /* Fallback (no debería ocurrir si el dataset se guardó bien) */
                            double exp_arr[1] = { expected };
                            for(int f_iter=0; f_iter<10; f_iter++) {
                                forward_propagation(red, rx_features);
                                backpropagation(red, exp_arr);
                            }
                        }
                        
                        /* Guardar memoria final */
                        guardar_pesos(red, "data/red_memoria.dat");

                        /* 3. Re-predecir para actualizar la UI instantaneamente */
                        forward_propagation(red, rx_features);
                        int ol = red->num_capas-1;
                        pred_out = red->capas[ol].neuronas[0].salida;
                        
                        add_msg("[APRENDIZAJE] Red re-entrenada exitosamente. Dataset actualizado.", 0);
                    }
                }
            }
        }

        EndDrawing();
    }

    if (rx_loaded) {
        UnloadTexture(current_rx_texture);
        UnloadImage(current_rx_img);
    }
    if (cfg)      liberar_config(cfg);
    if (ds)       liberar_dataset(ds);
    if (red)      liberar_red(red);
    if (mse_hist) free(mse_hist);

    CloseWindow();
}
