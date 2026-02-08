import asyncio
import json
import time
import urllib.parse
import uuid
from typing import Any, Dict, Optional

import httpx
import requests
import websocket

from .api import ComfyResponse


class _ComfyClientBase:
    def __init__(self, host: str = "127.0.0.1", port: int = 8188):
        self.host = host
        self.port = port
        self.client_id = str(uuid.uuid4())
        self.base_url = f"http://{host}:{port}"
        encoded_client_id = urllib.parse.quote(self.client_id)
        self.ws_url = f"ws://{host}:{port}/ws?clientId={encoded_client_id}"

    @staticmethod
    def _parse_response(response: httpx.Response) -> Any:
        try:
            return response.json()
        except ValueError:
            return response.content

    def _path(self, path: str) -> str:
        return f"{self.base_url}{path}"


class ComfyClient(_ComfyClientBase):
    def __init__(self, host: str = "127.0.0.1", port: int = 8188):
        super().__init__(host=host, port=port)
        self.client: Optional[httpx.Client] = None
        self.ws: Optional[websocket.WebSocket] = None

    def connect(self, connect_websocket: bool = False) -> None:
        self._ensure_http_client()
        if connect_websocket:
            self._ensure_ws()

    def close(self) -> None:
        if self.ws is not None:
            self.ws.close()
            self.ws = None
        if self.client is not None:
            self.client.close()
            self.client = None

    def _ensure_http_client(self) -> httpx.Client:
        if self.client is None:
            self.client = httpx.Client(timeout=30.0)
        return self.client

    def _ensure_ws(self) -> websocket.WebSocket:
        if self.ws is None:
            self.ws = websocket.WebSocket()
            self.ws.connect(self.ws_url)
        return self.ws

    def queue_prompt(self, prompt: Dict[str, Any]) -> ComfyResponse:
        payload = {"prompt": prompt, "client_id": self.client_id}
        response = self._ensure_http_client().post(self._path("/prompt"), json=payload)
        response.raise_for_status()
        result = response.json()
        return ComfyResponse(
            prompt_id=result.get("prompt_id"),
            number=result.get("number"),
            node_errors=result.get("node_errors"),
        )

    def get_queue(self) -> Dict[str, Any]:
        response = self._ensure_http_client().get(self._path("/prompt"))
        response.raise_for_status()
        return response.json()

    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        response = self._ensure_http_client().get(self._path(f"/history/{prompt_id}"))
        response.raise_for_status()
        return response.json()

    def get_all_history(self) -> Dict[str, Any]:
        response = self._ensure_http_client().get(self._path("/history"))
        response.raise_for_status()
        return response.json()

    def delete_history(self, prompt_id: str) -> Any:
        response = self._ensure_http_client().delete(self._path(f"/history/{prompt_id}"))
        response.raise_for_status()
        return self._parse_response(response)

    def clear_history(self) -> Any:
        response = self._ensure_http_client().delete(self._path("/history"))
        response.raise_for_status()
        return self._parse_response(response)

    def get_images(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        response = self._ensure_http_client().get(self._path("/view"), params=params)
        response.raise_for_status()
        return response.content

    def get_view_metadata(
        self,
        filename: str,
        subfolder: str = "",
        folder_type: str = "output",
    ) -> Dict[str, Any]:
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        response = self._ensure_http_client().get(self._path("/view_metadata"), params=params)
        response.raise_for_status()
        return response.json()

    def get_system_stats(self) -> Dict[str, Any]:
        response = self._ensure_http_client().get(self._path("/system_stats"))
        response.raise_for_status()
        return response.json()

    def get_extensions(self) -> Dict[str, Any]:
        response = self._ensure_http_client().get(self._path("/extensions"))
        response.raise_for_status()
        return response.json()

    def interrupt(self) -> Any:
        response = self._ensure_http_client().post(self._path("/interrupt"))
        response.raise_for_status()
        return self._parse_response(response)

    def clear_queue(self) -> Any:
        response = self._ensure_http_client().delete(self._path("/queue"))
        response.raise_for_status()
        return self._parse_response(response)

    def free(self, unload_models: bool = False, free_memory: bool = False) -> Any:
        payload = {"unload_models": unload_models, "free_memory": free_memory}
        response = self._ensure_http_client().post(self._path("/free"), json=payload)
        response.raise_for_status()
        return self._parse_response(response)

    def get_object_info(self, node_class: Optional[str] = None) -> Dict[str, Any]:
        path = "/object_info" if node_class is None else f"/object_info/{node_class}"
        response = self._ensure_http_client().get(self._path(path))
        response.raise_for_status()
        return response.json()

    def get_embeddings(self) -> list[str]:
        response = self._ensure_http_client().get(self._path("/embeddings"))
        response.raise_for_status()
        return response.json()

    def get_features(self) -> Dict[str, Any]:
        response = self._ensure_http_client().get(self._path("/features"))
        response.raise_for_status()
        return response.json()

    def get_models(self, folder: Optional[str] = None) -> list[str]:
        path = "/models" if folder is None else f"/models/{folder}"
        response = self._ensure_http_client().get(self._path(path))
        response.raise_for_status()
        return response.json()

    def get_workflow_templates(self) -> list[str]:
        response = self._ensure_http_client().get(self._path("/workflow_templates"))
        response.raise_for_status()
        return response.json()

    def get_users(self) -> list[Dict[str, Any]]:
        response = self._ensure_http_client().get(self._path("/users"))
        response.raise_for_status()
        return response.json()

    def create_user(self, username: str) -> Dict[str, Any]:
        response = self._ensure_http_client().post(self._path("/users"), json={"username": username})
        response.raise_for_status()
        return response.json()

    def get_userdata(self, file: str) -> bytes:
        quoted_file = urllib.parse.quote(file, safe="/")
        response = self._ensure_http_client().get(self._path(f"/userdata/{quoted_file}"))
        response.raise_for_status()
        return response.content

    def move_userdata(self, file: str, dest: str) -> Any:
        quoted_file = urllib.parse.quote(file, safe="/")
        quoted_dest = urllib.parse.quote(dest, safe="/")
        response = self._ensure_http_client().post(self._path(f"/userdata/{quoted_file}/move/{quoted_dest}"))
        response.raise_for_status()
        return self._parse_response(response)

    def upload_image(self, image_data: bytes, filename: str, overwrite: bool = False) -> Dict[str, Any]:
        response = requests.post(
            self._path("/upload/image"),
            files={"image": (filename, image_data)},
            data={"overwrite": "true" if overwrite else "false"},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def upload_mask(
        self,
        image_data: bytes,
        filename: str,
        original_ref: Dict[str, str],
        overwrite: bool = False,
        mask_type: str = "mask",
    ) -> Dict[str, Any]:
        response = requests.post(
            self._path("/upload/mask"),
            files={"image": (filename, image_data)},
            data={
                "original_ref": json.dumps(original_ref),
                "overwrite": "true" if overwrite else "false",
                "type": mask_type,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def wait_for_completion(self, prompt_id: str, timeout: int = 3600) -> Dict[str, Any]:
        ws = self._ensure_ws()
        start_time = time.time()

        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError("Timed out waiting for execution completion")

            try:
                out = ws.recv()
                if not isinstance(out, str):
                    continue
                message = json.loads(out)
                if message.get("type") != "executing":
                    continue
                data = message.get("data", {})
                if data.get("node") is None and data.get("prompt_id") == prompt_id:
                    return self.get_history(prompt_id)
            except websocket.WebSocketConnectionClosedException:
                self.ws = None
                ws = self._ensure_ws()
            except json.JSONDecodeError:
                continue


class AsyncComfyClient(_ComfyClientBase):
    def __init__(self, host: str = "127.0.0.1", port: int = 8188):
        super().__init__(host=host, port=port)
        self.client: Optional[httpx.AsyncClient] = None
        self.ws: Optional[websocket.WebSocket] = None

    async def connect(self, connect_websocket: bool = False) -> None:
        await self._ensure_http_client()
        if connect_websocket:
            await self._ensure_ws()

    async def close(self) -> None:
        if self.ws is not None:
            await asyncio.to_thread(self.ws.close)
            self.ws = None
        if self.client is not None:
            await self.client.aclose()
            self.client = None

    async def _ensure_http_client(self) -> httpx.AsyncClient:
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=30.0)
        return self.client

    async def _ensure_ws(self) -> websocket.WebSocket:
        if self.ws is None:
            ws = websocket.WebSocket()
            await asyncio.to_thread(ws.connect, self.ws_url)
            self.ws = ws
        return self.ws

    async def queue_prompt(self, prompt: Dict[str, Any]) -> ComfyResponse:
        payload = {"prompt": prompt, "client_id": self.client_id}
        client = await self._ensure_http_client()
        response = await client.post(self._path("/prompt"), json=payload)
        response.raise_for_status()
        result = response.json()
        return ComfyResponse(
            prompt_id=result.get("prompt_id"),
            number=result.get("number"),
            node_errors=result.get("node_errors"),
        )

    async def get_queue(self) -> Dict[str, Any]:
        client = await self._ensure_http_client()
        response = await client.get(self._path("/prompt"))
        response.raise_for_status()
        return response.json()

    async def get_history(self, prompt_id: str) -> Dict[str, Any]:
        client = await self._ensure_http_client()
        response = await client.get(self._path(f"/history/{prompt_id}"))
        response.raise_for_status()
        return response.json()

    async def get_all_history(self) -> Dict[str, Any]:
        client = await self._ensure_http_client()
        response = await client.get(self._path("/history"))
        response.raise_for_status()
        return response.json()

    async def delete_history(self, prompt_id: str) -> Any:
        client = await self._ensure_http_client()
        response = await client.delete(self._path(f"/history/{prompt_id}"))
        response.raise_for_status()
        return self._parse_response(response)

    async def clear_history(self) -> Any:
        client = await self._ensure_http_client()
        response = await client.delete(self._path("/history"))
        response.raise_for_status()
        return self._parse_response(response)

    async def get_images(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        client = await self._ensure_http_client()
        response = await client.get(self._path("/view"), params=params)
        response.raise_for_status()
        return response.content

    async def get_view_metadata(
        self,
        filename: str,
        subfolder: str = "",
        folder_type: str = "output",
    ) -> Dict[str, Any]:
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        client = await self._ensure_http_client()
        response = await client.get(self._path("/view_metadata"), params=params)
        response.raise_for_status()
        return response.json()

    async def get_system_stats(self) -> Dict[str, Any]:
        client = await self._ensure_http_client()
        response = await client.get(self._path("/system_stats"))
        response.raise_for_status()
        return response.json()

    async def get_extensions(self) -> Dict[str, Any]:
        client = await self._ensure_http_client()
        response = await client.get(self._path("/extensions"))
        response.raise_for_status()
        return response.json()

    async def interrupt(self) -> Any:
        client = await self._ensure_http_client()
        response = await client.post(self._path("/interrupt"))
        response.raise_for_status()
        return self._parse_response(response)

    async def clear_queue(self) -> Any:
        client = await self._ensure_http_client()
        response = await client.delete(self._path("/queue"))
        response.raise_for_status()
        return self._parse_response(response)

    async def free(self, unload_models: bool = False, free_memory: bool = False) -> Any:
        payload = {"unload_models": unload_models, "free_memory": free_memory}
        client = await self._ensure_http_client()
        response = await client.post(self._path("/free"), json=payload)
        response.raise_for_status()
        return self._parse_response(response)

    async def get_object_info(self, node_class: Optional[str] = None) -> Dict[str, Any]:
        path = "/object_info" if node_class is None else f"/object_info/{node_class}"
        client = await self._ensure_http_client()
        response = await client.get(self._path(path))
        response.raise_for_status()
        return response.json()

    async def get_embeddings(self) -> list[str]:
        client = await self._ensure_http_client()
        response = await client.get(self._path("/embeddings"))
        response.raise_for_status()
        return response.json()

    async def get_features(self) -> Dict[str, Any]:
        client = await self._ensure_http_client()
        response = await client.get(self._path("/features"))
        response.raise_for_status()
        return response.json()

    async def get_models(self, folder: Optional[str] = None) -> list[str]:
        path = "/models" if folder is None else f"/models/{folder}"
        client = await self._ensure_http_client()
        response = await client.get(self._path(path))
        response.raise_for_status()
        return response.json()

    async def get_workflow_templates(self) -> list[str]:
        client = await self._ensure_http_client()
        response = await client.get(self._path("/workflow_templates"))
        response.raise_for_status()
        return response.json()

    async def get_users(self) -> list[Dict[str, Any]]:
        client = await self._ensure_http_client()
        response = await client.get(self._path("/users"))
        response.raise_for_status()
        return response.json()

    async def create_user(self, username: str) -> Dict[str, Any]:
        client = await self._ensure_http_client()
        response = await client.post(self._path("/users"), json={"username": username})
        response.raise_for_status()
        return response.json()

    async def get_userdata(self, file: str) -> bytes:
        quoted_file = urllib.parse.quote(file, safe="/")
        client = await self._ensure_http_client()
        response = await client.get(self._path(f"/userdata/{quoted_file}"))
        response.raise_for_status()
        return response.content

    async def move_userdata(self, file: str, dest: str) -> Any:
        quoted_file = urllib.parse.quote(file, safe="/")
        quoted_dest = urllib.parse.quote(dest, safe="/")
        client = await self._ensure_http_client()
        response = await client.post(self._path(f"/userdata/{quoted_file}/move/{quoted_dest}"))
        response.raise_for_status()
        return self._parse_response(response)

    async def upload_image(self, image_data: bytes, filename: str, overwrite: bool = False) -> Dict[str, Any]:
        def _upload() -> Dict[str, Any]:
            response = requests.post(
                self._path("/upload/image"),
                files={"image": (filename, image_data)},
                data={"overwrite": "true" if overwrite else "false"},
                timeout=30,
            )
            response.raise_for_status()
            return response.json()

        return await asyncio.to_thread(_upload)

    async def upload_mask(
        self,
        image_data: bytes,
        filename: str,
        original_ref: Dict[str, str],
        overwrite: bool = False,
        mask_type: str = "mask",
    ) -> Dict[str, Any]:
        def _upload() -> Dict[str, Any]:
            response = requests.post(
                self._path("/upload/mask"),
                files={"image": (filename, image_data)},
                data={
                    "original_ref": json.dumps(original_ref),
                    "overwrite": "true" if overwrite else "false",
                    "type": mask_type,
                },
                timeout=30,
            )
            response.raise_for_status()
            return response.json()

        return await asyncio.to_thread(_upload)

    async def wait_for_completion(self, prompt_id: str, timeout: int = 3600) -> Dict[str, Any]:
        ws = await self._ensure_ws()
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError("Timed out waiting for execution completion")

            try:
                out = await asyncio.to_thread(ws.recv)
                if not isinstance(out, str):
                    continue
                message = json.loads(out)
                if message.get("type") != "executing":
                    continue
                data = message.get("data", {})
                if data.get("node") is None and data.get("prompt_id") == prompt_id:
                    return await self.get_history(prompt_id)
            except websocket.WebSocketConnectionClosedException:
                self.ws = None
                ws = await self._ensure_ws()
            except json.JSONDecodeError:
                continue

__all__ = ["ComfyClient", "AsyncComfyClient"]
