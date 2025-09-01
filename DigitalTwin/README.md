# DigitalTwin
1. Virtual(Unity) Digitaltwin
2. Real(Robot) DigitalTwin to Virtual



## Raspberry Pi 4 Remote Control from Windows

This guide explains how to set up **xrdp** on Raspberry Pi 4 to allow remote desktop access from Windows.  

---

### 1. Raspberry Pi Setup

1. Install **xrdp** on your Raspberry Pi:

```bash
sudo apt-get update##
sudo apt-get install -y xrdp
```

### 2. Window Setup

1. Open Remote Desktop Connection
2. If Black Screen Issue
(raspberry pi)

```bash
sudo vi /etc/xrdp/startwm.sh


unset DBUS_SESSION_BUS_ADDRESS
unset XDG_RUNTIME_DIR

test -x /etc/X11/Xsession && exec /etc/X11/Xsession
exec /bin/sh /etc/X11/Xsession

:wq
```