def execute_command(self, command, scope_name=None):
    """Komut yorumlayıcı"""
    if isinstance(command, dict):
        return None

    command = command.strip()
    if not command:
        return None
    command_upper = command.upper()

    if self.trace_mode:
        self.backtrace_logger.log(f"TRACE: Satır {self.program_counter + 1}: {command}")

    try:
        if command_upper.startswith("PRINT"):
            match = re.match(r"PRINT\s*(.+)?", command, re.IGNORECASE)
            if match:
                expr = match.group(1)
                if expr:
                    result = self.evaluate_expression(expr, scope_name)
                    print(result)
                else:
                    print()
            return None

        if command_upper.startswith("LET"):
            match = re.match(r"LET\s+(\w+)\s*=\s*(.+)", command, re.IGNORECASE)
            if match:
                var_name, expr = match.groups()
                value = self.evaluate_expression(expr, scope_name)
                if scope_name and scope_name in self.modules:
                    self.modules[scope_name]["variables"][var_name] = value
                else:
                    self.current_scope()[var_name] = value
            return None

        if command_upper.startswith("DIM"):
            match = re.match(r"DIM\s+(\w+)\s+AS\s+(\w+)(?:\s*\[\s*(.+)\s*\])?", command, re.IGNORECASE)
            if match:
                var_name, var_type, dims = match.groups()
                if dims:
                    dims = [self.evaluate_expression(d.strip(), scope_name) for d in dims.split(",")]
                    value = np.zeros(dims) if var_type.upper() in ["SINGLE", "DOUBLE"] else [None] * dims[0]
                else:
                    value = self.type_table[var_type.upper()]()
                if scope_name and scope_name in self.modules:
                    self.modules[scope_name]["variables"][var_name] = value
                else:
                    self.current_scope()[var_name] = value
            return None

        if command_upper.startswith("INPUT"):
            match = re.match(r"INPUT\s+(\w+)", command, re.IGNORECASE)
            if match:
                var_name = match.group(1)
                value = input()
                if scope_name and scope_name in self.modules:
                    self.modules[scope_name]["variables"][var_name] = value
                else:
                    self.current_scope()[var_name] = value
            return None

        if command_upper.startswith("IF"):
            condition = re.match(r"IF\s+(.+?)\s+THEN", command, re.IGNORECASE)
            if condition:
                result = self.evaluate_expression(condition.group(1), scope_name)
                self.if_stack.append(bool(result))
                if not result:
                    while self.program_counter < len(self.program):
                        next_cmd = self.program[self.program_counter][0].strip().upper()
                        if next_cmd.startswith("ELSE") or next_cmd == "ENDIF":
                            break
                        self.program_counter += 1
            return None

        if command_upper == "ELSE":
            if self.if_stack:
                if self.if_stack[-1]:
                    while self.program_counter < len(self.program):
                        if self.program[self.program_counter][0].strip().upper() == "ENDIF":
                            break
                        self.program_counter += 1
            return None

        if command_upper == "ENDIF":
            if self.if_stack:
                self.if_stack.pop()
            return None

        if command_upper.startswith("FOR"):
            match = re.match(r"FOR\s+(\w+)\s*=\s*(.+?)\s+TO\s+(.+?)(?:\s+STEP\s+(.+))?", command, re.IGNORECASE)
            if match:
                var_name, start, end, step = match.groups()
                start_val = self.evaluate_expression(start, scope_name)
                end_val = self.evaluate_expression(end, scope_name)
                step_val = self.evaluate_expression(step, scope_name) if step else 1
                self.loop_stack.append({
                    "var": var_name,
                    "current": start_val,
                    "end": end_val,
                    "step": step_val,
                    "start_pc": self.program_counter
                })
                if scope_name and scope_name in self.modules:
                    self.modules[scope_name]["variables"][var_name] = start_val
                else:
                    self.current_scope()[var_name] = start_val
            return None

        if command_upper == "NEXT":
            if self.loop_stack:
                loop_info = self.loop_stack[-1]
                var_name = loop_info["var"]
                current = loop_info["current"] + loop_info["step"]
                if (loop_info["step"] > 0 and current <= loop_info["end"]) or \
                   (loop_info["step"] < 0 and current >= loop_info["end"]):
                    loop_info["current"] = current
                    if scope_name and scope_name in self.modules:
                        self.modules[scope_name]["variables"][var_name] = current
                    else:
                        self.current_scope()[var_name] = current
                    self.program_counter = loop_info["start_pc"]
                else:
                    self.loop_stack.pop()
            return None

        if command_upper.startswith("WHILE"):
            condition = re.match(r"WHILE\s+(.+)", command, re.IGNORECASE)
            if condition:
                result = self.evaluate_expression(condition.group(1), scope_name)
                if result:
                    self.loop_stack.append({
                        "type": "while",
                        "condition": condition.group(1),
                        "start_pc": self.program_counter
                    })
                else:
                    nested = 1
                    while self.program_counter < len(self.program):
                        next_cmd = self.program[self.program_counter][0].strip().upper()
                        if next_cmd.startswith("WHILE"):
                            nested += 1
                        elif next_cmd == "WEND":
                            nested -= 1
                            if nested == 0:
                                break
                        self.program_counter += 1
            return None

        if command_upper == "WEND":
            if self.loop_stack and self.loop_stack[-1]["type"] == "while":
                loop_info = self.loop_stack[-1]
                if self.evaluate_expression(loop_info["condition"], scope_name):
                    self.program_counter = loop_info["start_pc"]
                else:
                    self.loop_stack.pop()
            return None

        if command_upper.startswith("DO"):
            self.loop_stack.append({
                "type": "do",
                "start_pc": self.program_counter
            })
            return None

        if command_upper.startswith("LOOP"):
            condition = re.match(r"LOOP\s+(?:UNTIL|WHILE)\s+(.+)", command, re.IGNORECASE)
            if condition and self.loop_stack and self.loop_stack[-1]["type"] == "do":
                is_until = "UNTIL" in command_upper
                result = self.evaluate_expression(condition.group(1), scope_name)
                if (is_until and not result) or (not is_until and result):
                    self.program_counter = self.loop_stack[-1]["start_pc"]
                else:
                    self.loop_stack.pop()
            return None

        if command_upper.startswith("SELECT CASE"):
            match = re.match(r"SELECT CASE\s+(.+)", command, re.IGNORECASE)
            if match:
                expr = match.group(1)
                value = self.evaluate_expression(expr, scope_name)
                self.select_stack.append({
                    "value": value,
                    "matched": False
                })
            return None

        if command_upper.startswith("CASE"):
            if self.select_stack:
                select_info = self.select_stack[-1]
                if not select_info["matched"]:
                    match = re.match(r"CASE\s+(.+)", command, re.IGNORECASE)
                    if match:
                        case_expr = match.group(1)
                        if case_expr.upper() == "ELSE":
                            select_info["matched"] = True
                        else:
                            case_value = self.evaluate_expression(case_expr, scope_name)
                            if case_value == select_info["value"]:
                                select_info["matched"] = True
                            else:
                                while self.program_counter < len(self.program):
                                    next_cmd = self.program[self.program_counter][0].strip().upper()
                                    if next_cmd.startswith("CASE") or next_cmd == "END SELECT":
                                        break
                                    self.program_counter += 1
            return None

        if command_upper == "END SELECT":
            if self.select_stack:
                self.select_stack.pop()
            return None

        if command_upper.startswith("GOTO"):
            label = command[5:].strip()
            if label in self.labels:
                self.program_counter = self.labels[label]
                return None
            else:
                raise PdsXException(f"Etiket bulunamadı: {label}")

        if command_upper.startswith("GOSUB"):
            label = command[6:].strip()
            if label in self.labels:
                self.call_stack.append(self.program_counter)
                self.program_counter = self.labels[label]
                return None
            else:
                raise PdsXException(f"Alt program etiketi bulunamadı: {label}")

        if command_upper == "RETURN":
            if self.call_stack:
                self.program_counter = self.call_stack.pop()
            return None

        if command_upper.startswith("ON ERROR"):
            if command_upper == "ON ERROR RESUME NEXT":
                self.error_handler = None
            else:
                match = re.match(r"ON ERROR GOTO\s+(.+)", command, re.IGNORECASE)
                if match:
                    label = match.group(1)
                    if label in self.labels:
                        self.error_handler = self.labels[label]
                    else:
                        raise PdsXException(f"Hata işleyici etiketi bulunamadı: {label}")
            return None

        if command_upper.startswith("SUB") or command_upper.startswith("FUNCTION"):
            is_function = command_upper.startswith("FUNCTION")
            match = re.match(r"(?:SUB|FUNCTION)\s+(\w+)(?:\((.*?)\))?", command, re.IGNORECASE)
            if match:
                name, params = match.groups()
                params = [p.strip() for p in params.split(",")] if params else []
                if is_function:
                    self.functions[name] = {"params": params, "body": [], "start_pc": self.program_counter}
                else:
                    self.subs[name] = {"params": params, "body": [], "start_pc": self.program_counter}
                while self.program_counter < len(self.program):
                    next_cmd = self.program[self.program_counter][0].strip().upper()
                    if next_cmd == f"END {'FUNCTION' if is_function else 'SUB'}":
                        break
                    self.program_counter += 1
            return None

        if command_upper.startswith("CALL "):
            match = re.match(r"CALL\s+(\w+)(?:\((.*?)\))?", command, re.IGNORECASE)
            if match:
                name, args = match.groups()
                args = [self.evaluate_expression(a.strip(), scope_name) for a in args.split(",")] if args else []
                if name in self.subs:
                    sub_info = self.subs[name]
                    new_scope = dict(zip(sub_info["params"], args))
                    self.local_scopes.append(new_scope)
                    self.call_stack.append(self.program_counter)
                    self.program_counter = sub_info["start_pc"]
                else:
                    raise PdsXException(f"Alt program bulunamadı: {name}")
            return None

        if "=" in command and not command_upper.startswith(("IF", "FOR", "SELECT")):
            var_name, expr = command.split("=", 1)
            var_name = var_name.strip()
            value = self.evaluate_expression(expr.strip(), scope_name)
            if scope_name and scope_name in self.modules:
                self.modules[scope_name]["variables"][var_name] = value
            else:
                self.current_scope()[var_name] = value
            return None

        if command_upper.startswith("LIBX."):
            parts = command.split(".")
            if len(parts) >= 3:
                module_name = parts[1].lower()
                func_name = parts[2].split("(")[0].upper()
                if module_name in self.libx_modules:
                    module = self.libx_modules[module_name]
                    if hasattr(module, func_name):
                        args_match = re.search(r"\((.*?)\)", command)
                        args = []
                        if args_match:
                            args = [self.evaluate_expression(a.strip(), scope_name) 
                                  for a in args_match.group(1).split(",")]
                        getattr(module, func_name)(*args)
                        return None

        if command_upper.startswith("BYTECODE."):
            parts = command[9:].split(".")
            if parts:
                op = parts[0].upper()
                if op in self.bytecode_opcodes:
                    args_match = re.search(r"\((.*?)\)", command)
                    args = []
                    if args_match:
                        args = [self.evaluate_expression(a.strip(), scope_name) 
                              for a in args_match.group(1).split(",")]
                    return self.bytecode_opcodes[op](*args)

        if command_upper.startswith(("SIMD.", "NEURAL.", "QUANTUM.", "GENETIC.", "BLOCKCHAIN.")):
            feature_type = command.split(".")[0].upper()
            if feature_type in self.bytecode_opcodes:
                op = command.split(".")[1].split("(")[0].upper()
                args_match = re.search(r"\((.*?)\)", command)
                args = []
                if args_match:
                    args = [self.evaluate_expression(a.strip(), scope_name) 
                          for a in args_match.group(1).split(",")]
                return self.bytecode_opcodes[feature_type][op](*args)

        # --- Dinamik modül komutları için ek kontrol ---
        try:
            from pdsXuv14 import parse_pdsx_command
            result = parse_pdsx_command(command)
            if result is not None:
                return result
        except Exception:
            pass

        raise PdsXException(f"Bilinmeyen komut: {command}")

    except Exception as e:
        if self.error_handler is not None:
            self.program_counter = self.error_handler
            return None
        raise PdsXException(f"Komut yürütme hatası: {str(e)}")
