import { readdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";

function groupEvaluations(evaluations) {
    const tools = new Map();

    for (const [model, modelEvaluations] of Object.entries(evaluations)) {
        for (const evaluation of modelEvaluations) {
            const toolId = evaluation.tool_uid;

            if (!tools.has(toolId)) {
                tools.set(toolId, {
                    tool_id: toolId,
                    evaluations: new Map()
                });
            }

            const tool = tools.get(toolId);

            if (!tool.evaluations.has(model)) {
                tool.evaluations.set(model, {
                    model,
                    with_source: null,
                    description_only: null
                });
            }

            const modelEvaluation = tool.evaluations.get(model);

            if (evaluation.scenario === "with_source") {
                modelEvaluation.with_source = evaluation.scores;
            } else if (evaluation.scenario === "description_only") {
                modelEvaluation.description_only = evaluation.scores;
            }
        }
    }

    const semiReuslt = Array.from(tools.values()).map(tool => ({
        tool_id: tool.tool_id,
        evaluations: Array.from(tool.evaluations.values())
    }));

    const result = semiReuslt.filter(a => a.evaluations.length > 1)
    console.log(`Há um total de ${result.length} tools em comum, e ${semiReuslt.length - result.length} em divergencia` )
    return result;
}

async function main() {
    const folder = "../data/evaluations/";

    const files = await readdir(folder);

    const evaluations = {};

    for (const file of files) {
        if (!file.endsWith(".jsonl")) continue;

        const filePath = path.join(folder, file);
        const content = await readFile(filePath, "utf-8");

        const objects = content
            .split(/\r?\n/)
            .filter(line => line.trim() !== "")
            .map(line => JSON.parse(line));

        const fileName = path.basename(file, ".jsonl");

        evaluations[fileName] = objects;
    }

    const finalResult = groupEvaluations(evaluations);

    await writeFile("file.json", JSON.stringify(finalResult))
}

main();