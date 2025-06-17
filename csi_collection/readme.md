https://github.com/nexmonster/nexmon_csi?tab=readme-ov-file (general information) <br>

https://github.com/nexmonster/nexmon_csi/discussions/2 (installation & usage) <br>

https://github.com/Gi-z/CSIKit (CSIKit) <br>

**Setup:**

192.168.3.1 (monitor)
192.168.1.1 (ap, ping receiver) -m 50:EB:F6:33:10:EC (eth6)
laptop (ping sender) (ping -i 0.01 -s 2600 192.168.1.1)

1. Reload router firmware:
```
/sbin/rmmod dhd
/sbin/insmod /jffs/dhd.ko
```

2. Bring up interfaces:
```
wl -i eth6 up
wl -i eth6 radio on
wl -i eth6 country BE
ifconfig eth6 up
```

3. Start CSI collection <br>
-c sets the WiFi channel to 36 (5 GHz band) with 80 MHz bandwidth. <br>
-s sets the sampling rate (500 Hz). <br>
```
/jffs/mcp -c 36/80 -C 1 -N 1 -m 50:EB:F6:33:10:EC
/jffs/nexutil -Ieth6 -s500 -b -l34 -v<key>
/usr/sbin/wl -i eth6 monitor 1
```

4. Save CSI data to file on laptop:
```
ssh admin@192.168.3.1 "/jffs/tcpdump -i eth6 dst port 5500 -w -" > csi_capture.pcap
```