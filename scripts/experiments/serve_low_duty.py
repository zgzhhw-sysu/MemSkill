"""Serial, loopback-only greedy Transformers inference with a synchronized per-token rest interval."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class DutyCycle:
    def __init__(self, fraction):
        self.fraction = fraction
        self.started = time.monotonic()

    def rest(self):
        torch.cuda.synchronize()
        elapsed = time.monotonic() - self.started
        time.sleep(elapsed * (1 / self.fraction - 1))
        self.started = time.monotonic()


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--model", required=True)
    args.add_argument("--port", type=int, required=True)
    args.add_argument("--duty", type=float, default=0.40)
    args.add_argument("--reserve-mib", type=int, default=2048,
                      help="Keep this much currently free GPU memory outside the PyTorch allocator cap")
    args = args.parse_args()
    if not 0 < args.duty <= 0.75 or torch.cuda.device_count() != 1:
        raise ValueError("Select one physical GPU and a duty fraction in (0, 0.75]")
    if args.reserve_mib < 0:
        raise ValueError("GPU memory reserve must be nonnegative")
    free, total = torch.cuda.mem_get_info()
    budget = min(int(0.76 * total), free - args.reserve_mib * 2**20)
    if budget <= 0:
        raise RuntimeError("Available GPU memory is below the requested reserve")
    torch.cuda.set_per_process_memory_fraction(budget / total)
    print(f"GPU allocator budget {budget // 2**20} MiB; free at startup {free // 2**20} MiB", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map={"": 0},
                                                 attn_implementation="sdpa", local_files_only=True).eval()
    stop_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, payload):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/v1/models":
                self.respond(200, {"object":"list", "data":[{"id":"llama3-8b","object":"model","owned_by":"local"}]})
            else:
                self.respond(404, {"error":"Unknown endpoint"})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.respond(404, {"error":"Unknown endpoint"})
                return
            try:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                limit = int(req["max_tokens"])
                if req.get("model") != "llama3-8b" or req.get("temperature") != 0 or not 1 <= limit <= 1024:
                    raise ValueError("This service supports greedy llama3-8b with 1–1024 output tokens")
                tokens = tokenizer.apply_chat_template(req["messages"], tokenize=True, add_generation_prompt=True,
                                                       return_tensors="pt").to(model.device)
                if tokens.shape[1] + limit > model.config.max_position_embeddings:
                    raise ValueError("Input and generation budget exceed the model context")
                with torch.inference_mode():
                    clock = DutyCycle(args.duty)
                    cache = None
                    for block in tokens.split(16, dim=1):
                        output = model(block, past_key_values=cache, use_cache=True, logits_to_keep=1)
                        cache = output.past_key_values
                        clock.rest()
                    ids = []
                    for step in range(limit):
                        token = output.logits[0, -1].argmax().view(1, 1)
                        token_id = int(token.item())
                        ids.append(token_id)
                        if token_id in stop_ids or step + 1 == limit:
                            break
                        output = model(token, past_key_values=cache, use_cache=True, logits_to_keep=1)
                        cache = output.past_key_values
                        clock.rest()
                answer = tokenizer.decode(ids, skip_special_tokens=True)
                self.respond(200, {"id":"local", "object":"chat.completion", "created":int(time.time()), "model":"llama3-8b",
                                    "choices":[{"index":0,"message":{"role":"assistant","content":answer},
                                                "finish_reason":"length" if len(ids) == limit else "stop"}],
                                    "usage":{"prompt_tokens":tokens.shape[1],"completion_tokens":len(ids),
                                             "total_tokens":tokens.shape[1]+len(ids)}})
            except (ValueError, KeyError) as exc:
                self.respond(400, {"error":str(exc)})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except RuntimeError as exc:
                self.respond(500, {"error":str(exc)})
    class SerialServer(HTTPServer):
        # The reader submits batches of up to 32 questions. Only the listening queue grows.
        request_queue_size = 64

    with SerialServer(("127.0.0.1", args.port), Handler) as server:
        print(f"Ready on {args.port}; duty={args.duty}; allocated={torch.cuda.memory_allocated()//2**20} MiB", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
