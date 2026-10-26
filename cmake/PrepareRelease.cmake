# SPDX-License-Identifier: Apache-2.0

execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" "${REPO_ROOT}/scripts/root.py" --revision
    WORKING_DIRECTORY "${REPO_ROOT}"
    RESULT_VARIABLE _revision_result
    OUTPUT_QUIET
    ERROR_VARIABLE _revision_error
)
if(NOT _revision_result EQUAL 0)
    message(FATAL_ERROR "A valid Git revision is required to build a release: ${_revision_error}")
endif()

execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" "${REPO_ROOT}/scripts/root.py"
    WORKING_DIRECTORY "${REPO_ROOT}"
    RESULT_VARIABLE _root_result
    ERROR_VARIABLE _root_error
)
if(NOT _root_result EQUAL 0)
    message(FATAL_ERROR "Failed to update .celune-root: ${_root_error}")
endif()

if(WIN32)
    execute_process(COMMAND taskkill /IM celune.exe /F OUTPUT_QUIET ERROR_QUIET)
    execute_process(COMMAND taskkill /IM celune-bin.exe /F OUTPUT_QUIET ERROR_QUIET)
else()
    find_program(_pkill NAMES pkill)
    if(_pkill)
        execute_process(COMMAND "${_pkill}" -TERM -x celune OUTPUT_QUIET ERROR_QUIET)
        execute_process(COMMAND "${_pkill}" -TERM -x celune-bin OUTPUT_QUIET ERROR_QUIET)
        execute_process(COMMAND "${CMAKE_COMMAND}" -E sleep 1)
        execute_process(COMMAND "${_pkill}" -KILL -x celune OUTPUT_QUIET ERROR_QUIET)
        execute_process(COMMAND "${_pkill}" -KILL -x celune-bin OUTPUT_QUIET ERROR_QUIET)
    endif()
endif()

file(MAKE_DIRECTORY "${OUTPUT_DIR}")
file(REMOVE "${OUTPUT_DIR}/celune" "${OUTPUT_DIR}/celune.exe")
file(REMOVE_RECURSE
    "${ARCHIVE_PATH}"
    "${OUTPUT_DIR}/celune.AppImage"
    "${OUTPUT_DIR}/celune-update.json"
    "${OUTPUT_DIR}/celune-update.sig"
    "${OUTPUT_DIR}/default_config.yaml"
    "${OUTPUT_DIR}/voices"
    "${OUTPUT_DIR}/resources"
    "${OUTPUT_DIR}/assets"
    "${OUTPUT_DIR}/vcruntime140.dll"
    "${OUTPUT_DIR}/celune-bin.cmd"
)
