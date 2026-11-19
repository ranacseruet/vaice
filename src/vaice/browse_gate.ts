/**
 * vaice gate: a Pi extension that decides, for every tool call the browsing agent makes, whether it runs, is refused,
 * or needs a person's yes first. vaice (the RPC client) answers the questions.
 *
 * The policy is the JSON in VAICE_GATE_POLICY ({"<tool or prefix*>": "run" | "ask" | "deny"}): the exact tool name
 * first, then the longest matching `prefix*` pattern, then "default" (ask when absent). "ask" sends a confirm dialog
 * to the RPC client and blocks the call unless it is confirmed; no answer within VAICE_GATE_TIMEOUT_MS counts as no.
 * A handler that throws also blocks the call (Pi's `tool_call` errors fail safe).
 *
 * The `vaice-gate` command is how vaice checks the extension loaded: Pi runs on without an extension that failed to
 * load, and the tools would then be ungated, so vaice refuses to start a task unless `get_commands` lists it.
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function (pi: ExtensionAPI) {
	const policy: Record<string, string> = JSON.parse(process.env.VAICE_GATE_POLICY ?? "{}");
	const timeout = Number(process.env.VAICE_GATE_TIMEOUT_MS ?? "60000");

	function tierOf(name: string): string {
		if (policy[name]) return policy[name];
		let best = "";
		for (const key of Object.keys(policy)) {
			if (key.endsWith("*") && name.startsWith(key.slice(0, -1)) && key.length > best.length) best = key;
		}
		return best ? policy[best] : (policy["default"] ?? "ask");
	}

	pi.registerCommand("vaice-gate", {
		description: "vaice's tool gate is active",
		handler: async () => {},
	});

	pi.on("tool_call", async (event, ctx) => {
		const tier = tierOf(event.toolName);
		if (tier === "run") return undefined;
		if (tier === "deny") return { block: true, reason: `${event.toolName} is not allowed` };
		if (!ctx.hasUI) return { block: true, reason: "no one to approve this" };
		const ok = await ctx.ui.confirm(event.toolName, JSON.stringify(event.input), { timeout });
		return ok ? undefined : { block: true, reason: `the user did not approve ${event.toolName}` };
	});
}
