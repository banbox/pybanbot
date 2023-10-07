#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : webserver.py
# Author: anyongjin
# Date  : 2023/9/6
from ipaddress import IPv4Address
from typing import Any

import orjson
import uvicorn
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from banbot.config import Config
from banbot.rpc import RPC, RPCException
from banbot.rpc.api.uvicorn_api import UvicornServer
from banbot.types import *
from banbot.util.common import logger


class MyJSONResponse(JSONResponse):
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        """
        Use rapidjson for responses
        Handles NaN and Inf / -Inf in a javascript way by default.
        """
        return orjson.dumps(content, option=orjson.OPT_SERIALIZE_NUMPY)


class ApiServer:
    obj = None
    _config: Config = {}
    rpc: Optional[RPC] = None

    def __new__(cls, *args, **kwargs):
        """
        This class is a singleton.
        We'll only have one instance of it around.
        """
        if ApiServer.obj is None:
            ApiServer.obj = object.__new__(cls)
        return ApiServer.obj

    def __init__(self, config: Config, standalone: bool = False) -> None:
        ApiServer._config = config
        if self.obj and standalone:
            return
        self._standalone: bool = standalone
        self._server = None

        api_config = self._config['api_server']

        self.app = FastAPI(title="Banbot API",
                           docs_url='/docs' if api_config.get('enable_openapi', False) else None,
                           redoc_url=None,
                           default_response_class=MyJSONResponse,
                           )
        self.configure_app(self.app, self._config)
        self.start_api()

    def add_rpc_handler(self, rpc: RPC):
        """
        Attach rpc handler
        """
        if not ApiServer.rpc:
            ApiServer.rpc = rpc
        else:
            # This should not happen assuming we didn't mess up.
            raise OperateError('RPC Handler already attached.')

    def configure_app(self, app: FastAPI, config):
        from banbot.rpc.api.auth import http_basic_or_jwt_token, router_login
        from banbot.rpc.api.api_v1 import router as api_v1
        from banbot.rpc.api.api_v1 import router_public as api_v1_public
        from banbot.rpc.api.base import DBSessionMiddleware

        auth_req = Depends(http_basic_or_jwt_token)
        app.include_router(api_v1_public, prefix="/api/v1")
        app.include_router(api_v1, prefix="/api/v1", dependencies=[auth_req])
        app.include_router(router_login, prefix="/api/v1", tags=["auth"])

        app.add_middleware(
            CORSMiddleware,
            allow_origins=config['api_server'].get('CORS_origins', []),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        app.add_middleware(DBSessionMiddleware)

        app.add_exception_handler(RPCException, self.handle_rpc_exception)

    def handle_rpc_exception(self, request, exc):
        logger.exception(f"API Error calling: {exc}")
        return JSONResponse(
            status_code=502,
            content={'error': f"Error querying {request.url.path}: {exc.message}"}
        )

    def start_api(self):
        """
        Start API ... should be run in thread.
        """
        rest_ip = self._config['api_server']['listen_ip_address']
        rest_port = self._config['api_server']['listen_port']

        logger.info(f'Starting HTTP Server at {rest_ip}:{rest_port}')
        if not IPv4Address(rest_ip).is_loopback:
            logger.warning("SECURITY WARNING - Local Rest Server listening to external connections")
            logger.warning("SECURITY WARNING - This is insecure please set to your loopback,"
                           "e.g 127.0.0.1 in config.json")

        if not self._config['api_server'].get('password'):
            logger.warning("SECURITY WARNING - No password for local REST Server defined. "
                           "Please make sure that this is intentional!")

        if (self._config['api_server'].get('jwt_secret_key', 'super-secret')
                in ('super-secret, somethingrandom')):
            logger.warning("SECURITY WARNING - `jwt_secret_key` seems to be default."
                           "Others may be able to log into your bot.")

        logger.info('Starting Local Rest Server.')
        verbosity = self._config['api_server'].get('verbosity', 'error')

        uvconfig = uvicorn.Config(self.app,
                                  port=rest_port,
                                  host=rest_ip,
                                  use_colors=False,
                                  log_config=None,
                                  access_log=True if verbosity != 'error' else False,
                                  ws_ping_interval=None  # We do this explicitly ourselves
                                  )
        try:
            self._server = UvicornServer(uvconfig)
            if self._standalone:
                self._server.run()
            else:
                self._server.run_in_thread()
        except Exception:
            logger.exception("Api server failed to start.")


def start_api(bot):
    '''
    启动RestApi线程
    '''
    config = bot.config
    if config.get('api_server', {}).get('enabled', False):
        logger.info('Enabling rpc.api_server')
        apiserver = ApiServer(config)
        apiserver.add_rpc_handler(RPC(bot))
