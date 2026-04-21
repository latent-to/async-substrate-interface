import subprocess
import time
import sys

import pytest

from async_substrate_interface import AsyncSubstrateInterface, SubstrateInterface
from async_substrate_interface.errors import MaxRetriesExceeded, StateDiscardedError
from async_substrate_interface.substrate_addons import (
    RetrySyncSubstrate,
    RetryAsyncSubstrate,
)
from tests.conftest import start_docker_container
from tests.helpers.settings import ARCHIVE_ENTRYPOINT, LATENT_LITE_ENTRYPOINT


def wait_for_output(process, target_string, timeout=60):
    """
    Wait for a specific string to appear in the subprocess stdout.

    Args:
        process: subprocess.Popen object
        target_string: String to wait for in stdout
        timeout: Maximum time to wait in seconds

    Returns:
        bool: True if string was found, False if timeout occurred
    """
    import time

    start_time = time.monotonic()

    # Make stdout non-blocking on Unix systems
    if sys.platform != "win32":
        import fcntl
        import os

        flags = fcntl.fcntl(process.stdout, fcntl.F_GETFL)
        fcntl.fcntl(process.stdout, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    buffer = ""
    while time.monotonic() - start_time < timeout:
        try:
            # Read available data
            chunk = process.stdout.read(1024)
            if chunk:
                chunk_str = chunk.decode("utf-8", errors="ignore")
                buffer += chunk_str
                print(chunk_str, end="", flush=True)  # Echo output for visibility

                if target_string in buffer:
                    return True
            else:
                # No data available, sleep briefly
                time.sleep(0.1)
        except (BlockingIOError, TypeError):
            # No data available yet
            time.sleep(0.1)

        # Check if process has terminated
        if process.poll() is not None:
            # Process ended, read remaining output
            remaining = process.stdout.read()
            if remaining:
                remaining_str = remaining.decode("utf-8", errors="ignore")
                print(remaining_str, end="", flush=True)
                if target_string in remaining_str:
                    return True
            return False

    return False


@pytest.fixture(scope="function")
def docker_containers():
    processes = (
        start_docker_container(9944, "9944"),
        start_docker_container(9945, "9945"),
    )
    try:
        yield processes

    finally:
        for process in processes:
            subprocess.run(["docker", "kill", process.name])
            process.process.kill()


@pytest.fixture(scope="function")
def single_local_chain():
    process = start_docker_container(9945, "9944")
    try:
        yield process
    finally:
        subprocess.run(["docker", "kill", process.name])
        process.process.kill()


def test_retry_sync_substrate(single_local_chain):
    # Wait for the Docker container to be ready
    if not wait_for_output(single_local_chain.process, "Imported #1", timeout=60):
        raise TimeoutError(
            "Docker container did not start properly - 'Imported #1' not found in output"
        )

    with RetrySyncSubstrate(
        single_local_chain.uri, fallback_chains=[LATENT_LITE_ENTRYPOINT]
    ) as substrate:
        for i in range(10):
            assert substrate.get_chain_head().startswith("0x")
            if i == 8:
                subprocess.run(["docker", "stop", single_local_chain.name])
            if i > 8:
                assert substrate.chain_endpoint == LATENT_LITE_ENTRYPOINT
            time.sleep(2)


def test_retry_sync_substrate_max_retries(docker_containers):
    # Wait for both Docker containers to be ready
    for i, container in enumerate(docker_containers):
        if not wait_for_output(container.process, "Imported #1", timeout=60):
            raise TimeoutError(
                f"Docker container {i} did not start properly - 'Imported #1' not found in output"
            )

    with RetrySyncSubstrate(
        docker_containers[0].uri, fallback_chains=[docker_containers[1].uri]
    ) as substrate:
        for i in range(5):
            assert substrate.get_chain_head().startswith("0x")
            if i == 2:
                subprocess.run(["docker", "pause", docker_containers[0].name])
            if i == 3:
                assert substrate.chain_endpoint == docker_containers[1].uri
            if i == 4:
                subprocess.run(["docker", "pause", docker_containers[1].name])
                with pytest.raises(MaxRetriesExceeded):
                    substrate.get_chain_head().startswith("0x")
            time.sleep(2)


def test_retry_sync_substrate_offline():
    with pytest.raises(ConnectionError):
        RetrySyncSubstrate(
            "ws://127.0.0.1:9944", fallback_chains=["ws://127.0.0.1:9945"]
        )


@pytest.mark.asyncio
async def test_retry_async_subtensor_archive_node():
    async with AsyncSubstrateInterface(LATENT_LITE_ENTRYPOINT) as substrate:
        current_block = await substrate.get_block_number()
        old_block = current_block - 1000
        with pytest.raises(StateDiscardedError):
            await substrate.get_block(block_number=old_block)
    async with RetryAsyncSubstrate(
        LATENT_LITE_ENTRYPOINT, archive_nodes=[ARCHIVE_ENTRYPOINT]
    ) as substrate:
        assert isinstance((await substrate.get_block(block_number=old_block)), dict)


def test_retry_sync_subtensor_archive_node():
    with SubstrateInterface(LATENT_LITE_ENTRYPOINT) as substrate:
        current_block = substrate.get_block_number()
        old_block = current_block - 1000
        with pytest.raises(StateDiscardedError):
            substrate.get_block(block_number=old_block)
    with RetrySyncSubstrate(
        LATENT_LITE_ENTRYPOINT, archive_nodes=[ARCHIVE_ENTRYPOINT]
    ) as substrate:
        assert isinstance((substrate.get_block(block_number=old_block)), dict)


@pytest.mark.asyncio
async def test_retry_async_substrate_runtime_call_with_keyword_args():
    """Test that runtime_call works with keyword arguments (parameter name conflict fix)."""
    async with RetryAsyncSubstrate(
        LATENT_LITE_ENTRYPOINT, retry_forever=True
    ) as substrate:
        # This should not raise TypeError due to parameter name conflict
        # The 'method' kwarg should not conflict with _retry's parameter
        result = await substrate.runtime_call(
            api="SwapRuntimeApi",
            method="current_alpha_price",
            params=[1],
            block_hash=None,
        )
        assert result is not None


def test_retry_sync_substrate_runtime_call_with_keyword_args():
    """Test that runtime_call works with keyword arguments (parameter name conflict fix)."""
    with RetrySyncSubstrate(LATENT_LITE_ENTRYPOINT, retry_forever=True) as substrate:
        # This should not raise TypeError due to parameter name conflict
        # The 'method' kwarg should not conflict with _retry's parameter
        result = substrate.runtime_call(
            api="SwapRuntimeApi",
            method="current_alpha_price",
            params=[1],
            block_hash=None,
        )
        assert result is not None
