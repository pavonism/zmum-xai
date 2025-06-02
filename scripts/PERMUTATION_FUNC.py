from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_permutation_importance(model, X_test, y_test, metric='accuracy', n_repeats=30, random_state=123,
                               top_n=10, xlim_max=None, threshold=0.001, plot_title="Most important features"):
    """
    Oblicza permutacyjną istotność cech i rysuje wykres typu lollipop.

    Parametry:
    - model: wytrenowany model klasyfikacyjny
    - X_test: zbiór cech testowych (DataFrame z nazwami kolumn)
    - y_test: etykiety testowe
    - metric: metryka oceny (np. 'accuracy', 'roc_auc')
    - n_repeats: liczba powtórzeń permutacji
    - random_state: wartość dla powtarzalności
    - top_n: liczba cech do pokazania (jeśli None: wszystkie powyżej threshold)
    - xlim_max: maksymalna wartość osi X
    - threshold: minimalna wartość istotności, by cecha została pokazana

    Zwraca:
    - importance_df: DataFrame z istotnością i błędem standardowym cech
    """

    # Obliczenie permutacyjnej istotności
    result = permutation_importance(model, X_test, y_test,
                                     n_repeats=n_repeats,
                                     random_state=123,
                                     scoring=metric)

    # Tworzymy DataFrame z wynikami
    importance_df = pd.DataFrame({
        'Feature': X_test.columns,
        'Mean Importance': result.importances_mean,
        'Std Importance': result.importances_std
    }).sort_values(by='Mean Importance', ascending=False).reset_index(drop=True)

    # Filtrowanie cech istotnych (powyżej progu)
    filtered = importance_df[importance_df['Mean Importance'] > threshold]

    # Automatyczne ustawienie top_n jeśli nie podano
    if top_n is None:
        top_n = len(filtered)

    # Wybór cech do pokazania
    top = filtered.head(top_n)

    # Dane do wykresu (odwracamy kolejność do wykresu od góry do dołu)
    y_pos = range(len(top))
    means = top['Mean Importance'][::-1]
    errors = top['Std Importance'][::-1]
    features = top['Feature'][::-1]

    # Wykres
    fig = plt.figure(figsize=(10, 0.5 * len(top) + 1))

    plt.errorbar(
        means,
        y_pos,
        xerr=errors,
        fmt='o',
        color='red',
        markersize=5,
        ecolor='black',
        elinewidth=1,
        alpha=0.9,
        capsize=3
    )

    plt.yticks(ticks=y_pos, labels=features)
    plt.xlabel("Drop in accuracy after permutation")
    plt.title(plot_title)
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    if xlim_max is not None:
        plt.xlim(0, xlim_max)

    plt.tight_layout()
    plt.show()

    return importance_df