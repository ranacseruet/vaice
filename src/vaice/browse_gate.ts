/**
 * vaice gate: a Pi extension that decides, for every tool call the browsing agent makes, whether it runs, is refused,
 * or needs a person's yes first. vaice (an ACP client, through pi-acp) answers the questions.
 *
 * The policy is the JSON in VAICE_GATE_POLICY ({"<tool or prefix*>": "run" | "ask" | "deny"}): the exact tool name
 * first, then the longest matching `prefix*` pattern, then "default" (ask when absent). "ask" sends a confirm dialog
 * to the client (pi-acp turns it into a permission request) and blocks the call unless it is confirmed; no answer
 * within VAICE_GATE_TIMEOUT_MS counts as no. A handler that throws also blocks the call (Pi's `tool_call` errors
 * fail safe).
 *
 * A gate that loads without hooking in would leave the tools ungated (one that throws while loading stops Pi from
 * starting), so vaice checks twice:
 * the `vaice-gate` command (vaice lists Pi's commands once at start-up and wants it there) and the notification
 * `vaice-gate <VAICE_GATE_NONCE>` this extension sends when each prompt starts (vaice wants it before the first tool
 * call of every task, or ends the agent).
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function (pi: ExtensionAPI) {
	const policy: Record<string, string> = JSON.parse(process.env.VAICE_GATE_POLICY ?? "{}");
	const timeout = Number(process.env.VAICE_GATE_TIMEOUT_MS ?? "60000");

	function tierOf(name: string): string {
		if (Object.prototype.hasOwnProperty.call(policy, name)) return policy[name];
		let best = "";
		for (const key of Object.keys(policy)) {
			if (key.endsWith("*") && name.startsWith(key.slice(0, -1)) && key.length > best.length) best = key;
		}
		return best ? policy[best] : (Object.prototype.hasOwnProperty.call(policy, "default") ? policy["default"] : "ask");
	}

	function navigationNeedsAsk(toolName: string, input: unknown): boolean {
		if (
			toolName !== "navigate_page" &&
			toolName !== "new_page" &&
			!toolName.endsWith("__navigate_page") &&
			!toolName.endsWith("__new_page")
		) {
			return false;
		}
		if (!input || typeof input !== "object") return false;
		const args = input as Record<string, unknown>;
		if ("initScript" in args && Boolean(args.initScript)) return true;
		if (typeof args.url === "string") {
			const trimmed = args.url.trim();
			if (trimmed !== "" && !/^https?:\/\//i.test(trimmed)) return true;
		}
		return false;
	}

	pi.registerCommand("vaice-gate", {
		description: "vaice's tool gate is active",
		handler: async () => {},
	});

	pi.on("before_agent_start", async (_event, ctx) => {
		if (ctx.hasUI) ctx.ui.notify(`vaice-gate ${process.env.VAICE_GATE_NONCE ?? ""}`, "info");
	});

	pi.on("tool_call", async (event, ctx) => {
		let tier = tierOf(event.toolName);
		if (tier === "run" && navigationNeedsAsk(event.toolName, event.input)) tier = "ask";
		if (tier === "run") return undefined;
		if (tier === "deny") return { block: true, reason: `${event.toolName} is not allowed` };
		if (!ctx.hasUI) return { block: true, reason: "no one to approve this" };
		const ok = await ctx.ui.confirm(event.toolName, JSON.stringify(event.input), { timeout });
		return ok ? undefined : { block: true, reason: `the user did not approve ${event.toolName}` };
	});
}
