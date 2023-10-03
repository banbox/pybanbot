#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : base.py
# Author: anyongjin
# Date  : 2023/9/9
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp
from banbot.storage import base as db_base
from fastapi import Request
from typing import Optional


class DBSessionMiddleware(BaseHTTPMiddleware):
    '''
    针对FastApi的sqlalchemy自动代理中间件
    app.add_middleware(DBSessionMiddleware)
    '''
    def __init__(
        self,
        app: ASGIApp,
        db_url: Optional[str] = None,
        iso_level: Optional[str] = None,
        session_args: Optional[dict] = None,
        commit_on_exit: bool = False,
    ):
        super().__init__(app)
        db_base.init_db(iso_level, db_url=db_url)
        self.session_args = session_args
        self.commit_on_exit = commit_on_exit

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        async with db_base.dba(session_args=self.session_args, commit_on_exit=self.commit_on_exit):
            response = await call_next(request)
        return response
