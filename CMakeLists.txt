# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.25)

cmake_policy(SET CMP0091 NEW)
cmake_policy(SET CMP0135 NEW)

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" _version_line REGEX "^version = ")
if(NOT _version_line)
    message(FATAL_ERROR "Could not determine the project version from pyproject.toml")
endif()
string(REGEX REPLACE "^version = \"([^\"]+)\".*$" "\\1" CELUNE_VERSION "${_version_line}")

project(CeluneNativeBuilder VERSION "${CELUNE_VERSION}" LANGUAGES C)
if(WIN32)
    enable_language(RC)
endif()

set(CELUNE_BUILD_MODE "unsigned" CACHE STRING "Release signature mode")
set_property(CACHE CELUNE_BUILD_MODE PROPERTY STRINGS unsigned signed)
if(NOT CELUNE_BUILD_MODE STREQUAL "unsigned" AND NOT CELUNE_BUILD_MODE STREQUAL "signed")
    message(FATAL_ERROR "CELUNE_BUILD_MODE must be unsigned or signed")
endif()

set(_require_signature_default OFF)
if("$ENV{CELUNE_REQUIRE_RELEASE_SIGNATURE}" STREQUAL "1")
    set(_require_signature_default ON)
endif()
option(CELUNE_REQUIRE_RELEASE_SIGNATURE "Require a signed release manifest" ${_require_signature_default})
if("$ENV{CELUNE_REQUIRE_RELEASE_SIGNATURE}" STREQUAL "1")
    set(CELUNE_REQUIRE_RELEASE_SIGNATURE ON)
endif()
if(CELUNE_BUILD_MODE STREQUAL "signed")
    set(CELUNE_REQUIRE_RELEASE_SIGNATURE ON)
endif()

set(CELUNE_RELEASE_SIGNING_KEY_FILE "$ENV{CELUNE_RELEASE_SIGNING_KEY_FILE}" CACHE FILEPATH "Release manifest signing key")
set(CELUNE_RELEASE_ROOT_KEY_FILE "$ENV{CELUNE_RELEASE_ROOT_KEY_FILE}" CACHE FILEPATH "Release keyring root key")
if(CELUNE_REQUIRE_RELEASE_SIGNATURE AND NOT CELUNE_RELEASE_SIGNING_KEY_FILE)
    if(WIN32 AND DEFINED ENV{LOCALAPPDATA})
        set(_default_signing_key "$ENV{LOCALAPPDATA}/Celune/keys/release-ed25519-private.pem")
    else()
        set(_default_signing_key "$ENV{HOME}/.celune/release-signing-key.pem")
    endif()
    set(CELUNE_RELEASE_SIGNING_KEY_FILE "${_default_signing_key}" CACHE FILEPATH "Release manifest signing key" FORCE)
endif()
if(WIN32 AND DEFINED ENV{LOCALAPPDATA})
    set(_default_root_key "$ENV{LOCALAPPDATA}/Celune/keys/release-keyring-root.pem")
else()
    set(_default_root_key "$ENV{HOME}/.celune/release-keyring-root.pem")
endif()
if(
    NOT CELUNE_RELEASE_ROOT_KEY_FILE
    OR (
        NOT "$ENV{CELUNE_RELEASE_ROOT_KEY_FILE}"
        AND CELUNE_RELEASE_SIGNING_KEY_FILE
        AND "${CELUNE_RELEASE_ROOT_KEY_FILE}" STREQUAL "${CELUNE_RELEASE_SIGNING_KEY_FILE}"
    )
)
    set(CELUNE_RELEASE_ROOT_KEY_FILE "${_default_root_key}" CACHE FILEPATH "Release keyring root key" FORCE)
endif()
if(CELUNE_RELEASE_SIGNING_KEY_FILE AND NOT EXISTS "${CELUNE_RELEASE_SIGNING_KEY_FILE}")
    message(FATAL_ERROR "The release signing key file was not found: ${CELUNE_RELEASE_SIGNING_KEY_FILE}")
endif()
if(CELUNE_REQUIRE_RELEASE_SIGNATURE AND NOT CELUNE_RELEASE_SIGNING_KEY_FILE)
    message(FATAL_ERROR "A release signing key is required when signature enforcement is enabled")
endif()
if((CELUNE_REQUIRE_RELEASE_SIGNATURE OR CELUNE_RELEASE_SIGNING_KEY_FILE) AND NOT EXISTS "${CELUNE_RELEASE_ROOT_KEY_FILE}")
    message(FATAL_ERROR "The release keyring root file was not found: ${CELUNE_RELEASE_ROOT_KEY_FILE}")
endif()
if(CELUNE_RELEASE_SIGNING_KEY_FILE)
    file(REAL_PATH "${CELUNE_RELEASE_SIGNING_KEY_FILE}" _release_signing_key_real_path)
    file(REAL_PATH "${CELUNE_RELEASE_ROOT_KEY_FILE}" _release_root_key_real_path)
    if("${_release_signing_key_real_path}" STREQUAL "${_release_root_key_real_path}")
        message(FATAL_ERROR "The release signing key and keyring root key must be separate files")
    endif()
endif()

if(WIN32)
    if(NOT MSVC)
        message(FATAL_ERROR "Configure the Windows build with the Visual Studio C compiler")
    endif()
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

    string(REGEX REPLACE "\\+.*$" "" _windows_version "${CELUNE_VERSION}")
    if(NOT _windows_version MATCHES "^[0-9]+(\\.[0-9]+)?(\\.[0-9]+)?(\\.[0-9]+)?$")
        message(FATAL_ERROR "Project version '${_windows_version}' is not a valid Windows version string")
    endif()
    string(REPLACE "." ";" _windows_version_components "${_windows_version}")
    list(LENGTH _windows_version_components _windows_version_component_count)
    while(_windows_version_component_count LESS 4)
        list(APPEND _windows_version_components 0)
        list(LENGTH _windows_version_components _windows_version_component_count)
    endwhile()
    list(JOIN _windows_version_components "," _windows_version_numbers)
    set(_celune_icon_path "${CMAKE_CURRENT_SOURCE_DIR}/resources/celune.ico")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/resources/celune.rc.in"
        "${CMAKE_CURRENT_BINARY_DIR}/celune.rc"
        @ONLY
    )
    set(_launcher_resource_file "${CMAKE_CURRENT_BINARY_DIR}/celune.rc")
    set_source_files_properties("${_launcher_resource_file}" PROPERTIES
        OBJECT_DEPENDS "${_celune_icon_path}"
    )
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(NOT CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        message(FATAL_ERROR "The Linux launcher requires GCC or Clang")
    endif()
else()
    message(FATAL_ERROR "The CMake release builder supports Windows and Linux")
endif()

find_program(UV_EXECUTABLE NAMES uv uv.exe REQUIRED)
find_program(PYTHON_EXECUTABLE NAMES python3 python REQUIRED)
if(NOT WIN32)
    find_program(CMAKE_MAKE_PROGRAM_FOR_SODIUM NAMES gmake make REQUIRED)
    find_program(APPIMAGETOOL_EXECUTABLE NAMES appimagetool REQUIRED)
endif()

include(ExternalProject)

set(_output_dir "${CMAKE_CURRENT_SOURCE_DIR}/bin")
if(CELUNE_BUILD_MODE STREQUAL "signed" OR CELUNE_RELEASE_SIGNING_KEY_FILE)
    if(WIN32)
        set(_archive_name "Celune-win-x64-signed.zip")
    else()
        set(_archive_name "Celune-linux-x64-signed.zip")
    endif()
elseif(WIN32)
    set(_archive_name "Celune-win-x64.zip")
else()
    set(_archive_name "Celune-linux-x64.zip")
endif()

add_custom_target(celune_prepare_release
    COMMAND "${CMAKE_COMMAND}"
        "-DREPO_ROOT=${CMAKE_CURRENT_SOURCE_DIR}"
        "-DOUTPUT_DIR=${_output_dir}"
        "-DARCHIVE_PATH=${_output_dir}/${_archive_name}"
        "-DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/PrepareRelease.cmake"
    VERBATIM
)

set(_libsodium_version "1.0.22")
set(_libsodium_url "https://download.libsodium.org/libsodium/releases/libsodium-${_libsodium_version}.tar.gz")
set(_libsodium_hash "SHA256=adbdd8f16149e81ac6078a03aca6fc03b592b89ef7b5ed83841c086191be3349")
set(_libsodium_install_dir "${CMAKE_BINARY_DIR}/libsodium")
set(_libsodium_include_dir "${_libsodium_install_dir}/include")
set(_libsodium_lib_dir "${_libsodium_install_dir}/lib")
file(MAKE_DIRECTORY "${_libsodium_include_dir}" "${_libsodium_lib_dir}")

if(WIN32)
    set(_libsodium_library "${_libsodium_lib_dir}/libsodium.lib")
    find_program(_vswhere NAMES vswhere.exe
        HINTS "$ENV{ProgramFiles\(x86\)}/Microsoft Visual Studio/Installer"
        NO_DEFAULT_PATH
    )
    if(_vswhere)
        execute_process(
            COMMAND "${_vswhere}" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64
                -find "MSBuild/**/Bin/MSBuild.exe"
            OUTPUT_VARIABLE _msbuild_path
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _vswhere_result
        )
        if(_vswhere_result EQUAL 0 AND _msbuild_path)
            string(REPLACE "\r\n" ";" _msbuild_candidates "${_msbuild_path}")
            list(GET _msbuild_candidates 0 MSBUILD_EXECUTABLE)
        endif()
    endif()
    if(NOT MSBUILD_EXECUTABLE)
        find_program(MSBUILD_EXECUTABLE NAMES MSBuild.exe msbuild.exe REQUIRED)
    endif()
    ExternalProject_Add(celune_libsodium_source
        URL "${_libsodium_url}"
        URL_HASH "${_libsodium_hash}"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        CONFIGURE_COMMAND ""
        BUILD_COMMAND "${MSBUILD_EXECUTABLE}"
            "<SOURCE_DIR>/builds/msvc/vs2022/libsodium.sln"
            /m:2 /v:minimal
            /p:Configuration=StaticRelease
            /p:Platform=x64
            "/p:OutDir=${_libsodium_lib_dir}/"
            "/p:IntDir=${CMAKE_BINARY_DIR}/libsodium-obj/"
        INSTALL_COMMAND "${CMAKE_COMMAND}"
            "-DLIBSODIUM_SOURCE_DIR=<SOURCE_DIR>"
            "-DLIBSODIUM_INSTALL_DIR=${_libsodium_install_dir}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/InstallLibsodiumWindows.cmake"
        BUILD_IN_SOURCE TRUE
        UPDATE_DISCONNECTED TRUE
    )
else()
    set(_libsodium_library "${_libsodium_lib_dir}/libsodium.a")
    ExternalProject_Add(celune_libsodium_source
        URL "${_libsodium_url}"
        URL_HASH "${_libsodium_hash}"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        CONFIGURE_COMMAND "${CMAKE_COMMAND}" -E env
            "CFLAGS=-O2 -DNDEBUG -fno-lto -fPIC"
            "<SOURCE_DIR>/configure"
            "--prefix=${_libsodium_install_dir}"
            --disable-shared --enable-static --with-pic
        BUILD_COMMAND "${CMAKE_MAKE_PROGRAM_FOR_SODIUM}" -j2
        INSTALL_COMMAND "${CMAKE_MAKE_PROGRAM_FOR_SODIUM}" install
        BUILD_IN_SOURCE TRUE
        UPDATE_DISCONNECTED TRUE
    )
endif()
add_dependencies(celune_libsodium_source celune_prepare_release)

add_library(celune_sodium STATIC IMPORTED GLOBAL)
set_target_properties(celune_sodium PROPERTIES
    IMPORTED_LOCATION "${_libsodium_library}"
    INTERFACE_INCLUDE_DIRECTORIES "${_libsodium_include_dir}"
)

set(_launcher_sources
    launcher/launcher.c
    launcher/signature.c
)
if(WIN32)
    list(APPEND _launcher_sources
        launcher/windows/runtime.c
        launcher/windows/terminal.c
        "${_launcher_resource_file}"
    )
else()
    list(APPEND _launcher_sources
        launcher/unix/runtime.c
        launcher/unix/terminal.c
    )
endif()

add_executable(celune_launcher ${_launcher_sources})
set_target_properties(celune_launcher PROPERTIES OUTPUT_NAME celune)
foreach(_configuration DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_property(TARGET celune_launcher PROPERTY "RUNTIME_OUTPUT_DIRECTORY_${_configuration}" "${_output_dir}")
endforeach()
target_include_directories(celune_launcher PRIVATE launcher)
target_link_libraries(celune_launcher PRIVATE celune_sodium)
add_dependencies(celune_launcher celune_libsodium_source)

add_executable(launcher_signature_test
    tests/launcher_signature_test.c
    launcher/signature.c
)
set_target_properties(launcher_signature_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/native-tests"
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
)
target_include_directories(launcher_signature_test PRIVATE launcher)
target_link_libraries(launcher_signature_test PRIVATE celune_sodium)
add_dependencies(launcher_signature_test celune_libsodium_source)

if(WIN32)
    target_compile_definitions(celune_launcher PRIVATE SODIUM_STATIC SODIUM_EXPORT=)
    target_compile_definitions(launcher_signature_test PRIVATE SODIUM_STATIC SODIUM_EXPORT=)
    target_compile_options(celune_launcher PRIVATE /O2 /GL /GS /guard:cf /W4 /DNDEBUG)
    target_link_libraries(celune_launcher PRIVATE bcrypt)
    target_link_options(celune_launcher PRIVATE /LTCG /OPT:REF /OPT:ICF /DYNAMICBASE /NXCOMPAT /GUARD:CF)
    target_compile_options(launcher_signature_test PRIVATE /O2 /W4)
    target_link_libraries(launcher_signature_test PRIVATE bcrypt)
else()
    find_package(Threads REQUIRED)
    target_compile_options(celune_launcher PRIVATE
        -O2 -Wall -Wextra -Wpedantic -DNDEBUG -D_FORTIFY_SOURCE=3
        -fstack-protector-strong -flto
    )
    target_link_options(celune_launcher PRIVATE -flto -s -Wl,-z,relro,-z,now)
    target_link_libraries(celune_launcher PRIVATE Threads::Threads)
    target_compile_options(launcher_signature_test PRIVATE -O2 -Wall -Wextra -Wpedantic)
    target_link_libraries(launcher_signature_test PRIVATE Threads::Threads)
endif()

add_custom_target(celune_native_verified
    COMMAND "$<TARGET_FILE:launcher_signature_test>"
    DEPENDS celune_launcher launcher_signature_test
    COMMENT "Running native launcher signature verification"
    VERBATIM
)

set(_nuitka_arguments
    --deployment
    --jobs=2
    --follow-import-to=celune
    --include-package-data=celune
    "--output-dir=${_output_dir}"
    --lto=yes
    "${CMAKE_CURRENT_SOURCE_DIR}/nuitka_main.py"
)
if(WIN32)
    list(PREPEND _nuitka_arguments
        --msvc=latest
        --include-windows-runtime-dlls=yes
        --windows-console-mode=force
        --product-name=Celune
        --file-description=Celune
        "--product-version=${_windows_version}"
        "--file-version=${_windows_version}"
        "--copyright=© celunah - Under Apache 2.0 license."
        --company-name=https://github.com/celunah
        "--output-filename=celune-bin.exe"
    )
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/resources/celune.ico")
        list(APPEND _nuitka_arguments "--windows-icon-from-ico=${CMAKE_CURRENT_SOURCE_DIR}/resources/celune.ico")
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/celune.exe")
        list(APPEND _nuitka_arguments "--windows-icon-from-exe=${CMAKE_CURRENT_SOURCE_DIR}/celune.exe")
    endif()
    set(_nuitka_environment
        "CL=/O2 /GL /GS /guard:cf /DNDEBUG"
        "_CL_=/link /LTCG /OPT:REF /OPT:ICF /DYNAMICBASE /NXCOMPAT"
    )
else()
    list(PREPEND _nuitka_arguments "--output-filename=celune-bin")
    set(_nuitka_environment)
endif()

if(WIN32)
    set(_python_core_output "${_output_dir}/celune-bin.exe")
else()
    set(_python_core_output "${_output_dir}/celune-bin")
endif()
file(GLOB_RECURSE _celune_core_sources CONFIGURE_DEPENDS LIST_DIRECTORIES false "${CMAKE_CURRENT_SOURCE_DIR}/celune/*")
list(FILTER _celune_core_sources EXCLUDE REGEX "/__pycache__/")
add_custom_command(
    OUTPUT "${_python_core_output}"
    COMMAND "${CMAKE_COMMAND}" -E env ${_nuitka_environment}
        "${UV_EXECUTABLE}" run --python 3.13 python -m nuitka ${_nuitka_arguments}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    DEPENDS
        ${_celune_core_sources}
        "${CMAKE_CURRENT_SOURCE_DIR}/nuitka_main.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml"
        "${CMAKE_CURRENT_SOURCE_DIR}/uv.lock"
    COMMENT "Compiling the Python core with Nuitka"
    VERBATIM
)
add_custom_target(celune_python_core DEPENDS "${_python_core_output}")
add_dependencies(celune_python_core celune_native_verified)

if(CELUNE_RELEASE_SIGNING_KEY_FILE)
    set(_signing_key_args "-DRELEASE_SIGNING_KEY_FILE=${CELUNE_RELEASE_SIGNING_KEY_FILE}")
else()
    set(_signing_key_args "-DRELEASE_SIGNING_KEY_FILE=")
endif()
if(CELUNE_REQUIRE_RELEASE_SIGNATURE)
    set(_require_signature_arg "-DREQUIRE_RELEASE_SIGNATURE=ON")
else()
    set(_require_signature_arg "-DREQUIRE_RELEASE_SIGNATURE=OFF")
endif()
set(_release_arguments)
if(NOT WIN32)
    list(APPEND _release_arguments "-DAPPIMAGETOOL_EXECUTABLE=${APPIMAGETOOL_EXECUTABLE}")
endif()

add_custom_target(celune_release
    COMMAND "${CMAKE_COMMAND}"
        "-DREPO_ROOT=${CMAKE_CURRENT_SOURCE_DIR}"
        "-DOUTPUT_DIR=${_output_dir}"
        "-DARCHIVE_NAME=${_archive_name}"
        "-DPROJECT_VERSION=${CELUNE_VERSION}"
        "-DPLATFORM_WINDOWS=${WIN32}"
        "-DPLATFORM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}"
        "-DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}"
        "-DUV_EXECUTABLE=${UV_EXECUTABLE}"
        "-DRELEASE_ROOT_KEY_FILE=${CELUNE_RELEASE_ROOT_KEY_FILE}"
        "${_signing_key_args}"
        "${_require_signature_arg}"
        ${_release_arguments}
        "-P" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/PackageRelease.cmake"
    DEPENDS celune_python_core
    COMMENT "Packaging the Celune release artifacts"
    VERBATIM
)
