# SPDX-License-Identifier: Apache-2.0

set(_headers_dir "${LIBSODIUM_SOURCE_DIR}/src/libsodium/include")
set(_archive "${LIBSODIUM_INSTALL_DIR}/lib/libsodium.lib")
if(NOT EXISTS "${_headers_dir}/sodium.h")
    message(FATAL_ERROR "The libsodium source build did not produce sodium.h")
endif()
if(NOT EXISTS "${_archive}")
    message(FATAL_ERROR "The libsodium source build did not produce libsodium.lib")
endif()
file(COPY "${_headers_dir}/" DESTINATION "${LIBSODIUM_INSTALL_DIR}/include")
