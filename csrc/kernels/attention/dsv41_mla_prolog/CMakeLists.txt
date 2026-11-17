# Author-owned options for the DS4.1 MLA prolog op package skeleton.
# These flags mirror the standalone CATLASS build; they are not build evidence
# for the in-repo op package. See the submission gap analysis.
function(flashinfer_configure_host target)
  target_compile_options(${target} PRIVATE -Wno-deprecated-declarations)
endfunction()

function(flashinfer_configure_kernel target)
  flashinfer_kernel_compile_options(${target}
    -Wno-macro-redefined
    -Wno-ignored-attributes
    -Wno-unused-value
    -mllvm -cce-aicore-record-overflow=true
    -mllvm -cce-aicore-dcci-insert-for-scalar=false)
endfunction()
