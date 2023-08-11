#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : fronts.py
# Author: anyongjin
# Date  : 2023/7/21
import datetime
from banbot.storage.base import *

min_date_time = datetime.datetime.utcfromtimestamp(0)


class Overlay(BaseDbModel):
    __tablename__ = 'overlay'

    id = Column(sa.Integer, primary_key=True)
    user = Column(sa.Integer, index=True)
    sid = Column(sa.Integer, index=True)
    start_ms = Column(sa.BIGINT, index=True)
    stop_ms = Column(sa.BIGINT, index=True)
    tf_msecs = Column(sa.Integer)
    update_at = Column(sa.DateTime, default=min_date_time)
    data = Column(sa.Text)

    @classmethod
    def get(cls, user_id: int, sid: int, start_ms: int, end_ms: int) -> List[dict]:
        import orjson
        sess = db.session
        sel_cols = [Overlay.id, Overlay.data]
        fts = [Overlay.user == user_id, Overlay.sid == sid, Overlay.start_ms >= start_ms, Overlay.stop_ms <= end_ms]
        overlays = list(sess.query(*sel_cols).filter(*fts).order_by(Overlay.id).all())
        result = []
        for row in overlays:
            if not row.data:
                continue
            try:
                data = orjson.loads(row.data)
            except ValueError:
                continue
            data['ban_id'] = row.id
            result.append(data)
        return result

    @classmethod
    def create(cls, user_id: int, sid: int, timeframe: str, data: dict):
        from banbot.util import btime
        import orjson
        olay_id = data.get('ban_id')
        sess = db.session
        points: List[dict] = data.get('points')
        if not points:
            logger.error(f'no points, skip overlay: {user_id}, {sid} {data}')
            return
        points = sorted(points, key=lambda x: x['timestamp'])
        start_ms, stop_ms = points[0]['timestamp'], points[-1]['timestamp']
        cur_time = btime.now()
        save_data = orjson.dumps(data).decode('utf-8')
        if olay_id:
            data.pop('ban_id')
            olay: Overlay = sess.query(Overlay).get(olay_id)
            if olay and olay.user == user_id:
                olay.start_ms = start_ms
                olay.stop_ms = stop_ms
                olay.update_at = cur_time
                olay.data = save_data
                sess.commit()
                return olay.id
        from banbot.exchange.exchange_utils import tf_to_secs
        tf_msecs = tf_to_secs(timeframe) * 1000
        olay = Overlay(user=user_id, sid=sid, start_ms=start_ms, stop_ms=stop_ms, tf_msecs=tf_msecs,
                       update_at=cur_time, data=save_data)
        sess.add(olay)
        sess.commit()
        return olay.id

    @classmethod
    def delete_by_id(cls, user_id: int, id_list: List[int]) -> int:
        sess = db.session
        fts = [Overlay.user == user_id, Overlay.id.in_(set(id_list))]
        count = sess.query(Overlay).filter(*fts).delete(synchronize_session=False)
        sess.commit()
        return count

    @classmethod
    def delete(cls, user_id: int, sid: int, timeframe: str, start_ms: int, stop_ms: int):
        from banbot.exchange.exchange_utils import tf_to_secs
        sess = db.session
        tf_msecs = tf_to_secs(timeframe) * 1000
        fts = [Overlay.user == user_id, Overlay.sid == sid, Overlay.tf_msecs == tf_msecs,
               Overlay.start_ms >= start_ms, Overlay.stop_ms <= stop_ms]
        count = sess.query(Overlay).filter(*fts).delete(synchronize_session=False)
        sess.commit()
        return count
