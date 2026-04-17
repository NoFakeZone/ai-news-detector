from features import FEATURE_MAP, TextStatsVariant

test_text = """
Sztuczne sieci neuronowe stanowią fundament współczesnej informatyki oraz systemów sztucznej inteligencji. 
Inspiracją dla ich powstania była budowa ludzkiego mózgu, a konkretnie sposób, w jaki neurony przesyłają sygnały elektryczne. 
W architekturze głębokiego uczenia wyróżniamy warstwy wejściowe, ukryte oraz wyjściowe. 
Każda warstwa składa się z węzłów, które przetwarzają dane przy użyciu wag i funkcji aktywacji. 
Obecnie potężne układy GPU pozwalają na trenowanie modeli o miliardach parametrów. 
Mimo że AI potrafi generować kod programistyczny, wciąż wymaga nadzoru człowieka. 
Programiści muszą dbać o optymalizację algorytmów, aby sieci działały wydajniej. 
Wydajność sieci neuronowych zależy od jakości danych treningowych oraz architektury samej SI.
"""


ttr_val = FEATURE_MAP[TextStatsVariant.TTR](test_text)
print(f"Klasyczny TTR:           {ttr_val:.4f}")

ttr_lem_val = FEATURE_MAP[TextStatsVariant.TTR_LEMATIZED](test_text)
print(f"Lematyzowany TTR:        {ttr_lem_val:.4f}")

cap_val = FEATURE_MAP[TextStatsVariant.CAPITAL_RATIO](test_text)
print(f"Capital Ratio:           {cap_val:.4f}")

asl_val = FEATURE_MAP[TextStatsVariant.AVG_SENTENCE_LEN](test_text)
print(f"Avg Sentence Len:        {asl_val:.2f}")