# Preserve the IRS device synchronization and compiler options.
function(flashinfer_configure_kernel target)
  flashinfer_kernel_compile_options(${target}
    --cce-auto-sync=off -Wno-deprecated-declarations -Werror
    -mllvm -cce-vf-remove-membar=false)
endfunction()
