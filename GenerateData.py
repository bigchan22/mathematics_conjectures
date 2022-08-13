import generate_data as gen

DIR_PATH = '/root/Hwang/mathematics_conjectures/Hwang/Data/N_7/'
# !mkdir {DIR_PATH}
gen.generate_data(DIR_PATH=DIR_PATH, input_N=7, primitive=True, connected=True, extended=False, UPTO_N=False)