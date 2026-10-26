# SPDX-License-Identifier: Apache-2.0

function(run_checked)
    execute_process(
        COMMAND ${ARGV}
        WORKING_DIRECTORY "${REPO_ROOT}"
        RESULT_VARIABLE _result
    )
    if(NOT _result EQUAL 0)
        string(JOIN " " _command ${ARGV})
        message(FATAL_ERROR "Release command failed (${_result}): ${_command}")
    endif()
endfunction()

if(PLATFORM_WINDOWS)
    set(_executable "celune.exe")
    set(_runtime "celune-bin.exe")
    set(_artifact "Celune-win-x64")
    set(_version "${PROJECT_VERSION}")
    string(REGEX REPLACE "\\+.*$" "" _version "${_version}")
    set(_archive_files celune.exe celune-bin.exe vcruntime140.dll celune-update.json)
    file(COPY_FILE "${REPO_ROOT}/resources/vcruntime140.dll" "${OUTPUT_DIR}/vcruntime140.dll" ONLY_IF_DIFFERENT)
else()
    set(_executable "celune")
    set(_runtime "celune-bin")
    set(_version "${PROJECT_VERSION}")
    string(TOLOWER "${PLATFORM_PROCESSOR}" _processor)
    if(_processor MATCHES "^(aarch64|arm64)$")
        set(_artifact_arch "aarch64")
    else()
        set(_artifact_arch "x86_64")
    endif()
    set(_artifact "Celune-linux-${_artifact_arch}")
    set(_archive_files celune celune-bin celune.AppImage celune-update.json)

    set(_appdir "${OUTPUT_DIR}/Celune.AppDir")
    file(REMOVE_RECURSE "${_appdir}")
    file(MAKE_DIRECTORY "${_appdir}")
    file(COPY_FILE "${REPO_ROOT}/Celune.AppDir/celune.desktop" "${_appdir}/celune.desktop")
    file(COPY_FILE "${REPO_ROOT}/Celune.AppDir/celune.png" "${_appdir}/celune.png")
    file(COPY_FILE "${OUTPUT_DIR}/celune" "${_appdir}/celune")
    file(CHMOD "${_appdir}/celune" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
    file(COPY_FILE "${REPO_ROOT}/cmake/AppRun.in" "${_appdir}/AppRun")
    file(CHMOD "${_appdir}/AppRun" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
    execute_process(COMMAND "${CMAKE_COMMAND}" -E create_symlink celune.png "${_appdir}/.DirIcon")
    if(NOT DEFINED APPIMAGETOOL_EXECUTABLE OR NOT EXISTS "${APPIMAGETOOL_EXECUTABLE}")
        message(FATAL_ERROR "appimagetool is required to create the Linux AppImage")
    endif()
    set(_appimage_arch "${_artifact_arch}")
    if(DEFINED ENV{ARCH} AND NOT "$ENV{ARCH}" STREQUAL "")
        string(TOLOWER "$ENV{ARCH}" _appimage_arch)
        if(_appimage_arch STREQUAL "amd64")
            set(_appimage_arch "x86_64")
        elseif(_appimage_arch STREQUAL "arm64")
            set(_appimage_arch "aarch64")
        endif()
    endif()
    run_checked("${CMAKE_COMMAND}" -E env "ARCH=${_appimage_arch}" "${APPIMAGETOOL_EXECUTABLE}"
        "${_appdir}" "${OUTPUT_DIR}/celune.AppImage")
    file(REMOVE_RECURSE "${_appdir}")
endif()

if(NOT EXISTS "${OUTPUT_DIR}/${_executable}" OR NOT EXISTS "${OUTPUT_DIR}/${_runtime}")
    message(FATAL_ERROR "The launcher and Nuitka runtime must exist before release packaging")
endif()
if(NOT EXISTS "${OUTPUT_DIR}/vcruntime140.dll" AND PLATFORM_WINDOWS)
    message(FATAL_ERROR "The Visual C++ runtime DLL is required for the Windows package")
endif()

execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" "${REPO_ROOT}/scripts/root.py" --revision
    WORKING_DIRECTORY "${REPO_ROOT}"
    OUTPUT_VARIABLE _revision
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _revision_result
)
if(NOT _revision_result EQUAL 0 OR _revision STREQUAL "")
    message(FATAL_ERROR "Could not read the Git revision for release metadata")
endif()

if(WIN32)
    set(_path_separator ";")
else()
    set(_path_separator ":")
endif()
set(_python_path "${REPO_ROOT}")
if(DEFINED ENV{PYTHONPATH} AND NOT "$ENV{PYTHONPATH}" STREQUAL "")
    string(APPEND _python_path "${_path_separator}$ENV{PYTHONPATH}")
endif()
set(_python_env
    "PYTHONPATH=${_python_path}"
    "UV_CACHE_DIR=${REPO_ROOT}/.uv-cache"
)

if(REQUIRE_RELEASE_SIGNATURE OR RELEASE_SIGNING_KEY_FILE)
    run_checked("${CMAKE_COMMAND}" -E env ${_python_env}
        "${UV_EXECUTABLE}" run python "${REPO_ROOT}/scripts/append_release_keyring.py"
        "${OUTPUT_DIR}/${_executable}" --root-key "${RELEASE_ROOT_KEY_FILE}")
endif()

set(_manifest_args
    run python "${REPO_ROOT}/scripts/write_update_manifest.py"
    --output-dir "${OUTPUT_DIR}"
    --version "${_version}"
    --revision "${_revision}"
    --artifact "${_artifact}"
    --file "${_executable}"
    --file "${_runtime}"
)
if(PLATFORM_WINDOWS)
    list(APPEND _manifest_args --file vcruntime140.dll)
else()
    list(APPEND _manifest_args --file celune.AppImage)
endif()
if(RELEASE_SIGNING_KEY_FILE)
    list(APPEND _manifest_args --signing-key "${RELEASE_SIGNING_KEY_FILE}")
endif()
if(REQUIRE_RELEASE_SIGNATURE)
    list(APPEND _manifest_args --require-signature)
endif()
run_checked("${CMAKE_COMMAND}" -E env ${_python_env} "${UV_EXECUTABLE}" ${_manifest_args})

if(EXISTS "${OUTPUT_DIR}/celune-update.sig")
    list(APPEND _archive_files celune-update.sig)
endif()
file(REMOVE "${OUTPUT_DIR}/${ARCHIVE_NAME}")
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E tar cf "${OUTPUT_DIR}/${ARCHIVE_NAME}" --format=zip ${_archive_files}
    WORKING_DIRECTORY "${OUTPUT_DIR}"
    RESULT_VARIABLE _archive_result
)
if(NOT _archive_result EQUAL 0)
    message(FATAL_ERROR "Could not create release archive ${ARCHIVE_NAME}")
endif()
file(REMOVE_RECURSE "${OUTPUT_DIR}/nuitka_main.build")
