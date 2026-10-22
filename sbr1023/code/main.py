"""
Clasificador difuso de riesgo climático usando una tabla BIOFAM.
Modelo de 2 entradas difusas y 1 salida crisp (sin desdifusificación).

Entradas:
- Temperatura (°C): FRIA, TEMPLADA, CALIENTE
- Humedad   (%):    BAJA, MEDIA, ALTA
Salida (etiqueta):
- BAJO, MEDIO, ALTO

"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt


TASK_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = TASK_DIR.parent
LIB_DIR = REPOSITORY_DIR / "lib"
sys.path.insert(0, str(LIB_DIR))

from syLibFuzzy import fuzzySET, fuzzyVAR, TRAPEZOIDAL
from syLibFuzzyPlot import plot_fuzzy_var, plot_fuzzy_var_value


def main():
    output_dir = TASK_DIR

    print("\n\nCLASIFICADOR DIFUSO DE RIESGO CLIMATICO USANDO BIOFAM")
    print("=" * 65)

    # ENTRADA 1: TEMPERATURA (0 a 45 °C, 3 conjuntos)
    in1 = fuzzyVAR("Temperatura", 0, 45, 3, "C")

    # FRIA: trapezoidal [0, 0, 10, 18]
    in1.fuzSet[0].setValues("FRIA",     TRAPEZOIDAL,  0,  0, 10, 18)
    # TEMPLADA: triangular (p1=p2=22), [15, 22, 22, 28]
    in1.fuzSet[1].setValues("TEMPLADA", TRAPEZOIDAL, 15, 22, 22, 28)
    # CALIENTE: trapezoidal [25, 35, 45, 45]
    in1.fuzSet[2].setValues("CALIENTE", TRAPEZOIDAL, 25, 35, 45, 45)

    # ENTRADA 2: HUMEDAD (0 a 100 %, 3 conjuntos)
    in2 = fuzzyVAR("Humedad", 0, 100, 3, "%")
    in2.fuzSet[0].setValues("BAJA",  TRAPEZOIDAL,  0,  0, 30, 50)
    in2.fuzSet[1].setValues("MEDIA", TRAPEZOIDAL, 40, 60, 60, 80)
    in2.fuzSet[2].setValues("ALTA",  TRAPEZOIDAL, 70, 85, 100, 100)

    # SALIDA: solo etiquetas (sin desdifusificación)
    Output_label = ["BAJO", "MEDIO", "ALTO"]

    # Mostrar variables
    print("\nENTRADA 1:")
    in1.showValues()
    print("\nENTRADA 2:")
    in2.showValues()
    print("\nETIQUETAS DE SALIDA:")
    print(Output_label)

    # TABLA BIOFAM
    # fam[r][c]:
    #   r = índice de Temperatura  (0=FRIA, 1=TEMPLADA, 2=CALIENTE)
    #   c = índice de Humedad       (0=BAJA, 1=MEDIA,   2=ALTA)
    #   fam[r][c] = índice de salida (0=BAJO, 1=MEDIO,  2=ALTO)
    fam = [
        [0, 0, 1],   # FRIA
        [0, 1, 2],   # TEMPLADA
        [1, 2, 2],   # CALIENTE
    ]

    # Visualizar la tabla
    print("\n\nTABLA BIOFAM (Temp renglones vs Humedad columnas):\n")
    header = "Temp\\Hum".ljust(12)
    for fs in in2.fuzSet:
        header += fs.label.center(10)
    print(header)
    print("-" * 45)
    for r in range(in1.numSets):
        row = in1.fuzSet[r].label.ljust(12)
        for c in range(in2.numSets):
            row += Output_label[fam[r][c]].center(10)
        print(row)
    print()

    # Mostrar las 9 reglas
    print("Las", in1.numSets * in2.numSets, "reglas son:\n")
    for r in range(in1.numSets):
        for c in range(in2.numSets):
            print(f"  R({r},{c}): SI {in1.label} es {in1.fuzSet[r].label} "
                  f"Y {in2.label} es {in2.fuzSet[c].label} "
                  f"-> salida = {Output_label[fam[r][c]]}")
    print()

    # CASOS DE PRUEBA
    casos = [
        (5,  15, "extremo seguro"),
        (40, 90, "extremo riesgoso"),
        (22, 60, "condición intermedia"),
        (27, 72, "caso ambiguo (compiten reglas)"),
        (0, 100, "extremos opuestos"),
    ]

    for (Vin1, Vin2, desc) in casos:
        print("\n" + "=" * 65)
        print(f"CASO: {desc}")
        print(f"Entradas: {in1.label} = {Vin1} {in1.units}, "
              f"{in2.label} = {Vin2} {in2.units}")
        print("=" * 65)

        # Membresías
        memIn1 = [s.membership(Vin1) for s in in1.fuzSet]
        memIn2 = [s.membership(Vin2) for s in in2.fuzSet]

        print("\nMembresías:")
        for i, mu in enumerate(memIn1):
            print(f"  u_{in1.fuzSet[i].label:<10}({Vin1:6.2f}) = {mu:.3f}")
        for i, mu in enumerate(memIn2):
            print(f"  u_{in2.fuzSet[i].label:<10}({Vin2:6.2f}) = {mu:.3f}")

        # Activación de reglas (min de membresías)
        act = [[0.0] * in2.numSets for _ in range(in1.numSets)]
        maxAct = -1.0
        rIdx = cIdx = -1
        for r in range(in1.numSets):
            for c in range(in2.numSets):
                act[r][c] = min(memIn1[r], memIn2[c])
                if act[r][c] > maxAct:
                    maxAct = act[r][c]
                    rIdx, cIdx = r, c

        # Matriz de activación
        print("\nMatriz de activación (min de membresías):")
        header2 = "         " + "".join(f"{fs.label:>10}" for fs in in2.fuzSet)
        print(header2)
        for r in range(in1.numSets):
            row = f"{in1.fuzSet[r].label:<9}"
            for c in range(in2.numSets):
                row += f"{act[r][c]:>10.3f}"
            print(row)

        # Resultado
        maxIdOut = fam[rIdx][cIdx]
        maxOutLabel = Output_label[maxIdOut]
        print(f"\nRegla ganadora: [{rIdx},{cIdx}] con activación = {maxAct:.3f}")
        print(f"  SI {in1.label} es {in1.fuzSet[rIdx].label} "
              f"Y {in2.label} es {in2.fuzSet[cIdx].label}")
        print(f"  ENTONCES riesgo = {maxOutLabel}")

    # GRAFICAS (para incluir en el reporte)
    print("\n\nGenerando gráficas de las variables lingüísticas...")

    fig1, ax1 = plot_fuzzy_var(in1)
    fig1.savefig(output_dir / "temp_var.png", dpi=150, bbox_inches="tight")

    fig2, ax2 = plot_fuzzy_var(in2)
    fig2.savefig(output_dir / "hum_var.png", dpi=150, bbox_inches="tight")

    # Con valor de prueba del caso ambiguo
    fig3, ax3 = plot_fuzzy_var_value(in1, 27)
    fig3.savefig(output_dir / "temp_var_value.png", dpi=150, bbox_inches="tight")

    fig4, ax4 = plot_fuzzy_var_value(in2, 72)
    fig4.savefig(output_dir / "hum_var_value.png", dpi=150, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()