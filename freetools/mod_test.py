import numpy as np
from matplotlib import pyplot as plt

import freetools as ft

snr = np.linspace(-4,24,100)
snr = 10**(snr/10)
x = 10*np.log10(snr)

bpsk = np.log10(ft.ber_bpsk(snr))
mpsk8 = np.log10(ft.ber_mpsk(snr, 8))
qpsk = np.log10(ft.ber_qpsk(snr))
mpsk16 = np.log10(ft.ber_mpsk(snr, 16))
mqam16 = np.log10(ft.ber_mqam(snr, 16))
mpsk32 = np.log10(ft.ber_mpsk(snr, 32))
mqam64 = np.log10(ft.ber_mqam(snr, 64))
dbpsk = np.log10(ft.ber_dbpsk(snr))
dqpsk = np.log10(ft.ber_dqpsk(snr))

f1 = plt.figure("Modulations")
ax = f1.add_subplot(111)
ax.plot(x, bpsk, color='cornflowerblue',label="BPSK, QPSK, 4-QAM")
ax.plot(x, dbpsk, color='deeppink',label="DBPSK")
ax.plot(x, dqpsk, color='black',label="DQPSK")
ax.plot(x, mpsk8, color='gold',label="8-PSK")
ax.plot(x, mqam16, color='yellowgreen',label="16-QAM")
ax.plot(x, mpsk16, color='aqua',label="16-PSK")
ax.plot(x, mqam64, color='sienna',label="64-QAM")
ax.plot(x, mpsk32, color='mediumorchid',label="32-PSK")

ax.set_title("log10(BER) per Eb/N0")
ax.set_ylabel("log10(BER)")
ax.set_xlabel("Eb/N0")
plt.legend()
plt.show()