cmd_/home/nancy/iir_test/DriverSrcLinux/Module.symvers := sed 's/\.ko$$/\.o/' /home/nancy/iir_test/DriverSrcLinux/modules.order | scripts/mod/modpost -m -a  -o /home/nancy/iir_test/DriverSrcLinux/Module.symvers -e -i Module.symvers   -T -
