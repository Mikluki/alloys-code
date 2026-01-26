from vsf.calculators.custom import Mace_mpa_0

calculator = Mace_mpa_0()
calculator.initialize()
print(calculator.energy_source.value)
print(calculator.get_model_info())
