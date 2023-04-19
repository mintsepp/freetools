import numpy as np
from matplotlib import pyplot as plt

import freetools as ft

snr = np.linspace(-4,24,100)
snr = 10**(snr/10)
x = 10*np.log10(snr)

ook = np.log10(ft.ber_ook(snr))
bpsk = np.log10(ft.ber_bpsk(snr))
mpsk8 = np.log10(ft.ber_mpsk(snr, 8))
qpsk = np.log10(ft.ber_qpsk(snr))
mpsk16 = np.log10(ft.ber_mpsk(snr, 16))
mqam16 = np.log10(ft.ber_mqam(snr, 16))
mpsk32 = np.log10(ft.ber_mpsk(snr, 32))
mqam64 = np.log10(ft.ber_mqam(snr, 64))
dbpsk = np.log10(ft.ber_dbpsk(snr))
dqpsk = np.log10(ft.ber_dqpsk(snr))

f1 = plt.figure("SpectralEfficiency")
ax = f1.add_subplot(111)
ax.plot(16.60,	1.00, 'o',color='slategray',label="OOK")
ax.plot(10.60,	1.00, 'o',color='cornflowerblue',label="BPSK")
ax.plot(10.60,	2.00, 'o',color='darkblue',label="QPSK")
ax.plot(14.00,	3.00, 'o',color='gold',label="8-PSK")
ax.plot(18.50,	4.00, 'o',color='aqua',label="16-PSK")
ax.plot(23.40,	5.00, 'o',color='mediumorchid',label="32-PSK")
ax.plot(10.60,	2.00, '*',color='skyblue',label="4-QAM")
ax.plot(14.40,	4.00, 'o',color='yellowgreen',label="16-QAM")
ax.plot(18.80,	6.00, 'o',color='sienna',label="64-QAM")
ax.plot(11.20,	1.00, 'o',color='deeppink',label="DBPSK")
ax.plot(12.90,	2.00, 'o',color='black',label="DQPSK")


ax.set_title("Spectral efficiency per Eb/N0 for BER = 10^-6")
ax.set_ylabel("spectral efficiency")
ax.set_xlabel("Eb/N0 [dB]")
plt.grid()
plt.legend()
plt.show()