cmd_/home/nancy/new/DriverSrcLinux/modules.order := {   echo /home/nancy/new/DriverSrcLinux/MSRdrv.ko; :; } | awk '!x[$$0]++' - > /home/nancy/new/DriverSrcLinux/modules.order
