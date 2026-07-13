/**
 * @file main.c
 * @brief Punto de entrada de SINA-VISUAL.
 */
#include <stdlib.h>
#include <time.h>

/* app.c expone la función de arranque directamente */
void iniciar_interfaz_grafica(void);

int main(void) {
    srand((unsigned int)time(NULL));
    iniciar_interfaz_grafica();
    return 0;
}
