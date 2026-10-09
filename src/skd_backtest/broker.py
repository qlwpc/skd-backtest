"""A-share order execution through the fixed cache and cost-quote protocol."""

from bisect import bisect_left
from dataclasses import replace
from math import isclose, isfinite, nextafter

import pandas as pd

from .config import BacktestConfig
from .contracts import CostQuote, CostRequest, ExecutionResult, Topic
from .runtime_cache import CacheView
from .schemas import LOCK_COLUMNS, RESULT_COLUMNS, STATE_COLUMNS


_CASH_TOLERANCE = 1e-8
_MAX_BUY_QUOTES = 64


def _finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _required_number(value, name):
    number = _finite_number(value)
    if number is None:
        raise ValueError(f"{name} must be finite")
    return number


def _flag(value):
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    return bool(value)


def _same(left, right):
    try:
        left_missing, right_missing = pd.isna(left), pd.isna(right)
        if left_missing and right_missing:
            return True
        if left_missing or right_missing:
            return False
    except (TypeError, ValueError):
        pass
    return left == right


def _positive_price(value):
    number = _finite_number(value)
    return number if number is not None and number > 0 else None


def _share_count(value, name):
    number = _required_number(value, name)
    if number < 0 or not number.is_integer():
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(number)


class Broker:
    def __init__(self, config: BacktestConfig):
        self.config = config

    def start_day(self, *, date: str, cache: CacheView) -> None:
        dates = cache.read(Topic.RUN_CALENDAR).trading_dates
        index = bisect_left(dates, date)
        account = (cache.read(Topic.ACCOUNT_CLOSE, dates[index - 1]).account if index
                   else cache.read(Topic.ACCOUNT_INITIAL).account)
        if account.price_mode != self.config.price_mode:
            raise ValueError("settlement account price mode does not match the run")

        if account.price_mode == "raw_price" and not account.locked_lots.empty:
            locked = account.locked_lots
            due = [
                isinstance(unlock_date, str) and unlock_date <= date
                for _, _, unlock_date, _ in locked.itertuples(index=False, name=None)
            ]
            if any(due):
                releases = {}
                for row, is_due in zip(locked.itertuples(index=False, name=None), due):
                    if is_due:
                        code, shares = row[0], _share_count(row[1], "locked shares")
                        releases[code] = releases.get(code, 0) + shares

                positions = account.positions
                row_by_code = {
                    row[0]: index
                    for index, row in zip(positions.index, positions.itertuples(index=False, name=None))
                }
                if len(row_by_code) != len(positions):
                    raise ValueError("raw account positions contain duplicate codes")
                settled_positions = positions.copy()
                for code, shares in releases.items():
                    if code not in row_by_code:
                        raise ValueError(f"locked shares have no position: {code}")
                    row_index = row_by_code[code]
                    total = _share_count(settled_positions.at[row_index, "total_shares"], "total_shares")
                    sellable = _share_count(settled_positions.at[row_index, "sellable_shares"], "sellable_shares")
                    if sellable + shares > total:
                        raise ValueError(f"unlock exceeds total shares: {code}")
                    settled_positions.at[row_index, "sellable_shares"] = sellable + shares
                settled_lots = locked.loc[[not value for value in due]].copy()
                account = replace(account, positions=settled_positions, locked_lots=settled_lots)
                cache.log(level="DEBUG", message="settled due share lots",
                          details={"released_lots": sum(due), "released_shares": sum(releases.values())})
            else:
                cache.log(level="DEBUG", message="no share lots due for settlement")
        else:
            cache.log(level="DEBUG", message="no share lots due for settlement")

        cache.publish(Topic.ACCOUNT_SETTLED, date, account)

    def execute(self, *, date: str, market: pd.DataFrame, cache: CacheView):
        snapshot = cache.read(Topic.ACCOUNT_OPEN, date)
        account = snapshot.account
        if not cache.contains(Topic.SIGNAL_TARGETS, date):
            cache.publish(Topic.EXECUTION_DAY, date, ExecutionResult(
                date, account,
                pd.DataFrame(columns=RESULT_COLUMNS["orders"]),
                pd.DataFrame(columns=RESULT_COLUMNS["trades"]),
                0.0, 0.0, None,
            ))
            return
        if account.price_mode != self.config.price_mode:
            raise ValueError("open account price mode does not match the run")

        equity = _required_number(snapshot.portfolio_value, "open portfolio value")
        if equity < 0:
            raise ValueError("open portfolio value cannot be negative")

        plan = cache.read(Topic.SIGNAL_TARGETS, date)
        signal_date = plan.signal_date
        target_weights = {}
        if plan.execution_date != date:
            raise ValueError("target plan is not scheduled for this execution date")
        for row in zip(*(plan.weights[name].to_numpy() for name in plan.weights)):
            row_signal, row_execution, code, _, _, raw_weight = row
            if row_signal != plan.signal_date or row_execution != date:
                raise ValueError("target row dates do not match its plan")
            if code in target_weights:
                raise ValueError(f"target plan contains duplicate code: {code}")
            weight = _required_number(raw_weight, f"target weight for {code}")
            if weight < 0:
                raise ValueError(f"target weight cannot be negative: {code}")
            target_weights[code] = weight
        if sum(target_weights.values()) > 1.0 + 1e-9:
            raise ValueError("target weights cannot exceed 1")

        open_column = "raw_open" if account.price_mode == "raw_price" else "adjusted_open"
        required_market = ("code", open_column, "is_suspended", "is_missing", "upper_limit", "lower_limit")
        try:
            for name in required_market:
                market.columns.get_loc(name)
        except KeyError as exc:
            raise ValueError(f"open market is missing required field: {exc.args[0]}") from exc
        market_columns = {name: index for index, name in enumerate(required_market)}
        market_rows = {}
        for row in zip(*(market[name].to_numpy() for name in required_market)):
            code = row[market_columns["code"]]
            if code in market_rows:
                raise ValueError(f"open market contains duplicate code: {code}")
            market_rows[code] = row

        values_by_code = {}
        for row in zip(*(snapshot.values[name].to_numpy() for name in snapshot.values)):
            code, price, price_date, market_value = row
            if code in values_by_code:
                raise ValueError(f"open values contain duplicate code: {code}")
            values_by_code[code] = (price, price_date, market_value)

        columns = STATE_COLUMNS[account.price_mode]
        source_positions = {}
        for row in zip(*(account.positions[name].to_numpy() for name in account.positions)):
            code = row[0]
            if code in source_positions:
                raise ValueError(f"account positions contain duplicate code: {code}")
            source_positions[code] = row
        changed_positions = {}

        if account.price_mode == "adjusted_return":
            for code, row in source_positions.items():
                if code not in values_by_code:
                    raise ValueError(f"open valuation is missing for held security: {code}")
                open_price, price_date, market_value = values_by_code[code]
                open_price = _positive_price(open_price)
                market_value = _required_number(market_value, f"open value for {code}")
                if open_price is None or market_value < 0:
                    raise ValueError(f"open valuation is invalid for held security: {code}")
                updated = dict(zip(columns, row))
                if (not _same(updated["position_value"], market_value)
                        or not _same(updated["reference_price"], open_price)
                        or not _same(updated["reference_date"], price_date)):
                    updated.update(position_value=market_value, reference_price=open_price,
                                   reference_date=price_date)
                    changed_positions[code] = updated

        market_indexes = market_columns

        def market_info(code):
            row = market_rows.get(code)
            if row is None:
                return None, None, "MISSING_OPEN", None, None
            price = _positive_price(row[market_indexes[open_column]])
            suspended = _flag(row[market_indexes["is_suspended"]])
            missing = _flag(row[market_indexes["is_missing"]])
            if suspended is True:
                reason = "SUSPENDED"
            elif missing is not False or suspended is None or price is None:
                reason = "MISSING_OPEN"
            else:
                reason = None
            if reason is None:
                upper = _finite_number(row[market_indexes["upper_limit"]])
                lower = _finite_number(row[market_indexes["lower_limit"]])
            else:
                upper = lower = None
            return row, price, reason, upper, lower

        def current_position(code):
            if code in changed_positions:
                return changed_positions[code]
            row = source_positions.get(code)
            return dict(zip(columns, row)) if row is not None else None

        intents = []
        codes = sorted(set(source_positions) | set(target_weights))
        for code in codes:
            weight = target_weights.get(code, 0.0)
            target_value = equity * weight
            position = current_position(code)
            row, open_price, market_reason, upper, lower = market_info(code)
            if account.price_mode == "adjusted_return":
                current_value = 0.0 if position is None else _required_number(
                    values_by_code[code][2], f"open value for {code}")
                difference = target_value - current_value
                if abs(difference) <= max(1e-9, abs(equity) * 1e-12):
                    continue
                side = "BUY" if difference > 0 else "SELL"
                reason = market_reason
                if reason is None and (upper is None or lower is None or lower <= 0 or upper <= lower):
                    reason = "MISSING_PRICE_LIMIT"
                if reason is None and side == "BUY" and (
                        open_price >= upper or isclose(open_price, upper, rel_tol=1e-12, abs_tol=0.0)):
                    reason = "LIMIT_UP"
                elif reason is None and side == "SELL" and (
                        open_price <= lower or isclose(open_price, lower, rel_tol=1e-12, abs_tol=0.0)):
                    reason = "LIMIT_DOWN"
                intents.append({
                    "code": code, "side": side, "shares": None,
                    "amount": abs(difference), "requested_value": abs(difference),
                    "base_price": None, "market_reason": reason,
                })
                continue

            total = 0 if position is None else _share_count(position["total_shares"], "total_shares")
            if weight == 0:
                desired = 0
            else:
                sizing_price = open_price
                if sizing_price is None and code in values_by_code:
                    sizing_price = _positive_price(values_by_code[code][0])
                desired = None if sizing_price is None else int((target_value / sizing_price) // 100) * 100
            if desired is None:
                current_value = 0.0 if code not in values_by_code else _required_number(
                    values_by_code[code][2], f"open value for {code}")
                difference = target_value - current_value
                if abs(difference) <= max(1e-9, abs(equity) * 1e-12):
                    continue
                side = "BUY" if difference > 0 else "SELL"
                requested_shares = None
                requested_value = abs(difference)
            else:
                difference = desired - total
                if difference > 0:
                    requested_shares = (difference // 100) * 100
                    if requested_shares == 0:
                        continue
                    side = "BUY"
                elif difference < 0:
                    requested_shares = -difference
                    side = "SELL"
                else:
                    continue
                sizing_price = open_price
                if sizing_price is None and code in values_by_code:
                    sizing_price = _positive_price(values_by_code[code][0])
                if sizing_price is None:
                    requested_value = (0.0 if side == "SELL" else target_value)
                else:
                    requested_value = requested_shares * sizing_price
            if (market_reason is None and account.price_mode == "raw_price"
                    and (upper is None or lower is None or upper <= 0 or lower <= 0 or upper < lower)):
                raise ValueError(f"raw open limits are invalid for tradable security: {code}")
            reason = market_reason
            if reason is None and side == "BUY" and upper is not None and open_price >= upper:
                reason = "LIMIT_UP"
            elif reason is None and side == "SELL" and lower is not None and open_price <= lower:
                reason = "LIMIT_DOWN"
            intents.append({
                "code": code, "side": side, "shares": requested_shares,
                "amount": None, "requested_value": requested_value,
                "base_price": open_price, "market_reason": reason,
            })

        order_rows, trade_rows, new_lots = [], [], []
        cash = _required_number(account.cash, "cash")
        if cash < 0:
            raise ValueError("cash cannot be negative")
        trade_value = total_cost = 0.0
        request_id = 0

        def request_quote(intent, shares, amount):
            nonlocal request_id
            request_id += 1
            order_id = f"{date}:{intent['side']}:{intent['code']}"
            if account.price_mode == "raw_price":
                request = CostRequest(request_id, order_id, date, intent["side"], account.price_mode,
                                      intent["base_price"], shares, None)
            else:
                request = CostRequest(request_id, order_id, date, intent["side"], account.price_mode,
                                      None, None, amount)
            key = (date, request_id)
            cache.publish(Topic.COST_REQUEST, key, request)
            yield request_id
            try:
                quote = cache.read(Topic.COST_RESULT, key)
                self._validate_quote(request, quote)
                return quote
            finally:
                if cache.contains(Topic.COST_RESULT, key):
                    cache.finish_quote(date=date, request_id=request_id)

        def append_order(intent, *, status, reason, filled_shares=None, filled_value=0.0, order_id=None):
            order_rows.append({
                "date": date, "code": intent["code"], "side": intent["side"],
                "requested_shares": intent["shares"], "status": status,
                "reject_reason": reason,
                "order_id": order_id or f"{date}:{intent['side']}:{intent['code']}",
                "signal_date": signal_date, "requested_value": intent["requested_value"],
                "filled_shares": filled_shares, "filled_value": filled_value,
            })

        def update_position(intent, quote, filled_shares):
            code = intent["code"]
            position = current_position(code)
            if account.price_mode == "adjusted_return":
                current = 0.0 if position is None else _required_number(position["position_value"], "position value")
                value = current + quote.position_value if intent["side"] == "BUY" else current - quote.position_value
                if value < -_CASH_TOLERANCE:
                    raise ValueError(f"trade exceeds adjusted position value: {code}")
                value = max(0.0, value)
                if value <= _CASH_TOLERANCE:
                    changed_positions[code] = None
                else:
                    if position is None:
                        price = _positive_price(market_rows[code][market_indexes[open_column]])
                        price_date = date
                        position = {"code": code, "position_value": 0.0,
                                     "reference_price": price, "reference_date": price_date}
                    position["position_value"] = value
                    changed_positions[code] = position
                return

            total = 0 if position is None else _share_count(position["total_shares"], "total_shares")
            sellable = 0 if position is None else _share_count(position["sellable_shares"], "sellable_shares")
            if intent["side"] == "BUY":
                total += filled_shares
                price = intent["base_price"]
                if position is None:
                    position = {"code": code, "total_shares": 0, "sellable_shares": 0,
                                "reference_price": price, "reference_date": date}
                else:
                    position["reference_price"] = price
                    position["reference_date"] = date
                new_lots.append((code, filled_shares, next_trading_date(), "T+1"))
            else:
                if filled_shares > sellable:
                    raise ValueError(f"sell exceeds sellable shares: {code}")
                total -= filled_shares
                sellable -= filled_shares
                if total < 0 or sellable < 0:
                    raise ValueError(f"sell exceeds total shares: {code}")
            if total == 0:
                changed_positions[code] = None
            else:
                position.update(total_shares=total, sellable_shares=sellable)
                changed_positions[code] = position

        calendar_dates = None

        def next_trading_date():
            nonlocal calendar_dates
            if calendar_dates is None:
                calendar_dates = cache.read(Topic.RUN_CALENDAR).trading_dates
            index = bisect_left(calendar_dates, date)
            if index >= len(calendar_dates) or calendar_dates[index] != date:
                raise ValueError(f"execution date is absent from the trading calendar: {date}")
            return calendar_dates[index + 1] if index + 1 < len(calendar_dates) else None

        for intent in sorted((item for item in intents if item["side"] == "SELL"), key=lambda item: item["code"]):
            code = intent["code"]
            order_id = f"{date}:SELL:{code}"
            if intent["market_reason"] is not None:
                append_order(intent, status="REJECTED", reason=intent["market_reason"], order_id=order_id)
                continue
            requested_shares = intent["shares"]
            if account.price_mode == "raw_price":
                position = current_position(code)
                sellable = 0 if position is None else _share_count(position["sellable_shares"], "sellable_shares")
                if requested_shares is None or requested_shares <= 0 or sellable <= 0:
                    append_order(intent, status="REJECTED", reason="T1_NOT_SELLABLE", order_id=order_id)
                    continue
                filled_shares = min(requested_shares, sellable)
                request_amount = None
            else:
                filled_shares, request_amount = None, intent["amount"]

            quote = yield from request_quote(intent, filled_shares, request_amount)
            next_cash = cash + quote.cash_delta
            if next_cash < 0.0:
                append_order(intent, status="REJECTED", reason="INSUFFICIENT_CASH", order_id=order_id)
                continue
            cash = max(0.0, next_cash)
            update_position(intent, quote, filled_shares)
            partial_reason = (
                "T1_NOT_SELLABLE"
                if account.price_mode == "raw_price" and filled_shares < requested_shares
                else None
            )
            append_order(
                intent,
                status="PARTIALLY_FILLED" if partial_reason else "FILLED",
                reason=partial_reason, filled_shares=filled_shares,
                filled_value=quote.position_value, order_id=order_id,
            )
            trade_rows.append(self._trade_row(date, signal_date, intent, order_id, quote, filled_shares))
            trade_value += quote.trade_value
            total_cost += quote.total_cost

        for intent in sorted((item for item in intents if item["side"] == "BUY"), key=lambda item: item["code"]):
            order_id = f"{date}:BUY:{intent['code']}"
            if intent["market_reason"] is not None:
                append_order(intent, status="REJECTED", reason=intent["market_reason"], order_id=order_id)
                continue
            if cash <= 0:
                append_order(intent, status="REJECTED", reason="INSUFFICIENT_CASH", order_id=order_id)
                continue

            original_shares = intent["shares"]
            original_amount = intent["amount"]
            proposed_shares, proposed_amount = original_shares, original_amount
            accepted = None
            previous_adjusted_quote = None
            for _ in range(_MAX_BUY_QUOTES):
                if account.price_mode == "raw_price" and (proposed_shares is None or proposed_shares < 100):
                    break
                if account.price_mode == "adjusted_return" and (proposed_amount is None or proposed_amount <= 0):
                    break
                quote = yield from request_quote(intent, proposed_shares, proposed_amount)
                next_cash = cash + quote.cash_delta
                if next_cash >= 0.0:
                    accepted = quote, proposed_shares, proposed_amount
                    break

                outflow = -quote.cash_delta
                if outflow <= 0:
                    raise ValueError("buy quote must reduce cash")
                if account.price_mode == "raw_price":
                    new_shares = int((proposed_shares * max(0.0, cash / outflow)) // 100) * 100
                    if new_shares >= proposed_shares:
                        new_shares = proposed_shares - 100
                    if new_shares < 100:
                        break
                    proposed_shares = new_shares
                else:
                    scaled = proposed_amount * max(0.0, cash / outflow)
                    if previous_adjusted_quote is not None:
                        previous_amount, previous_outflow = previous_adjusted_quote
                        amount_delta = proposed_amount - previous_amount
                        outflow_delta = outflow - previous_outflow
                        if amount_delta and outflow_delta:
                            slope = outflow_delta / amount_delta
                            candidate = proposed_amount + (cash - outflow) / slope if slope > 0 else None
                        else:
                            candidate = None
                        if candidate is not None and isfinite(candidate) and 0 < candidate < proposed_amount:
                            scaled = nextafter(candidate, 0.0)
                        else:
                            scaled = proposed_amount / 2
                    elif scaled >= proposed_amount:
                        scaled = nextafter(proposed_amount, 0.0)
                    if scaled <= 0 or scaled >= proposed_amount:
                        break
                    previous_adjusted_quote = (proposed_amount, outflow)
                    proposed_amount = scaled

            if accepted is None:
                append_order(intent, status="REJECTED", reason="INSUFFICIENT_CASH", order_id=order_id)
                continue
            quote, filled_shares, filled_amount = accepted
            next_cash = cash + quote.cash_delta
            cash = max(0.0, next_cash)
            update_position(intent, quote, filled_shares)
            filled_value = quote.position_value
            if account.price_mode == "raw_price":
                partial = filled_shares < original_shares
                partial_reason = "INSUFFICIENT_CASH" if partial else None
            else:
                partial = filled_amount < original_amount and not isclose(
                    filled_amount, original_amount, rel_tol=1e-12, abs_tol=1e-9)
                partial_reason = "INSUFFICIENT_CASH" if partial else None
            append_order(
                intent, status="PARTIALLY_FILLED" if partial else "FILLED",
                reason=partial_reason, filled_shares=filled_shares,
                filled_value=filled_value, order_id=order_id,
            )
            trade_rows.append(self._trade_row(date, signal_date, intent, order_id, quote, filled_shares))
            trade_value += quote.trade_value
            total_cost += quote.total_cost

        positions = self._materialize_positions(account.positions, columns, source_positions, changed_positions)
        locked_lots = self._materialize_lots(account.locked_lots, new_lots)
        if cash != account.cash or positions is not account.positions or locked_lots is not account.locked_lots:
            account = replace(account, cash=cash, positions=positions, locked_lots=locked_lots)

        result = ExecutionResult(
            date, account,
            pd.DataFrame(order_rows, columns=RESULT_COLUMNS["orders"]),
            pd.DataFrame(trade_rows, columns=RESULT_COLUMNS["trades"]),
            trade_value, total_cost, signal_date,
        )
        cache.publish(Topic.EXECUTION_DAY, date, result)
        cache.log(level="DEBUG", message="completed daily order execution",
                  details={"orders": len(order_rows), "trades": len(trade_rows)})
        yield from ()

    @staticmethod
    def _validate_quote(request: CostRequest, quote: CostQuote) -> None:
        for name in ("request_id", "order_id", "date", "side", "price_mode"):
            if getattr(quote, name) != getattr(request, name):
                raise ValueError("cost quote does not match its request")
        for name in ("position_value", "trade_value", "commission", "stamp_tax", "other_cost", "total_cost"):
            value = _required_number(getattr(quote, name), f"quote {name}")
            if value < 0:
                raise ValueError(f"quote {name} cannot be negative")
        if not isclose(quote.total_cost, quote.commission + quote.stamp_tax + quote.other_cost,
                       rel_tol=1e-9, abs_tol=1e-8):
            raise ValueError("cost quote total_cost does not equal its explicit fees")
        if quote.trade_value <= 0 or quote.position_value <= 0:
            raise ValueError("a nonzero order requires positive quoted values")
        expected_cash = (
            -(quote.trade_value + quote.total_cost)
            if request.side == "BUY"
            else quote.trade_value - quote.total_cost
        )
        if not isclose(quote.cash_delta, expected_cash, rel_tol=1e-9, abs_tol=1e-8):
            raise ValueError("cost quote cash_delta does not match its trade and fees")
        if request.price_mode == "raw_price":
            if quote.execution_price is None or _positive_price(quote.execution_price) is None:
                raise ValueError("raw cost quote requires a positive execution price")
            expected_position = request.base_price * request.shares
            expected_trade = quote.execution_price * request.shares
            if (not isclose(quote.position_value, expected_position, rel_tol=1e-9, abs_tol=1e-8)
                    or not isclose(quote.trade_value, expected_trade, rel_tol=1e-9, abs_tol=1e-8)):
                raise ValueError("raw cost quote values do not match its shares")
        else:
            if quote.execution_price is not None or not isclose(
                    quote.position_value, request.position_value, rel_tol=1e-9, abs_tol=1e-8):
                raise ValueError("adjusted cost quote must preserve its requested asset amount")

    @staticmethod
    def _trade_row(date, signal_date, intent, order_id, quote, filled_shares):
        return {
            "date": date, "code": intent["code"], "side": intent["side"],
            "shares": filled_shares, "price": quote.execution_price,
            "trade_value": quote.trade_value, "commission": quote.commission,
            "stamp_tax": quote.stamp_tax, "other_cost": quote.other_cost,
            "total_cost": quote.total_cost, "order_id": order_id,
            "signal_date": signal_date, "position_value": quote.position_value,
        }

    @staticmethod
    def _materialize_positions(original, columns, source, changes):
        if not changes:
            return original
        records = []
        for code in sorted(set(source) | set(changes)):
            if code in changes:
                row = changes[code]
            else:
                row = dict(zip(columns, source[code]))
            if row is not None:
                records.append([row[column] for column in columns])
        return pd.DataFrame(records, columns=columns)

    @staticmethod
    def _materialize_lots(original, additions):
        if not additions:
            return original
        rows = list(original.itertuples(index=False, name=None))
        rows.extend(additions)
        rows.sort(key=lambda row: (row[0], not isinstance(row[2], str), row[2] if isinstance(row[2], str) else "", row[3]))
        return pd.DataFrame(rows, columns=LOCK_COLUMNS)
