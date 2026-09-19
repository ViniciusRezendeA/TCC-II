import { readdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";

type Tool = {
    tool_id: string;
    evaluations: ModelEvaluation[];
};

type ModelEvaluation = {
    model: string;
    with_source?: Evaluation;
    description_only?: Evaluation;
};

type Score = {
    score: number;
    reasoning: string;
};

type Evaluation = {
    purpose: Score;
    guidelines: Score;
    limitations: Score;
    parameter_explanation: Score;
    length_completeness: Score;
    examples: Score;
};

const metrics: (keyof Evaluation)[] = [
    "purpose",
    "guidelines",
    "limitations",
    "parameter_explanation",
    "length_completeness",
    "examples"
];

async function main() {
    const file = await readFile("file.json");
    const content: Tool[] = JSON.parse(file);

    const filtered: Tool[] = content
        .map(tool => {
            const evaluations = tool.evaluations
    .filter(model =>
        model.with_source != null &&
        model.description_only != null
    )
    .map(modelEvaluation => {
        const withSource = modelEvaluation.with_source!;
        const descriptionOnly = modelEvaluation.description_only!;

        const differences = Object.fromEntries(
            metrics
                .filter(metric =>
                    withSource[metric].score !==
                    descriptionOnly[metric].score
                )
                .map(metric => [
                    metric,
                    {
                        with_source: withSource[metric],
                        description_only: descriptionOnly[metric],
                        difference:
                            withSource[metric].score -
                            descriptionOnly[metric].score
                    }
                ])
        );

        const totalDifference = Object.values(differences)
            .reduce(
                (sum, metric) => sum + metric.difference,
                0
            );

        return {
            differences,
            totalDifference
        };
    })
    .filter(model =>
        Object.keys(model.differences).length > 0
    )
    .sort((a, b) =>
        b.totalDifference - a.totalDifference
    );

            return {
                ...tool,
                evaluations,
                totalDifference: evaluations.reduce(
                    (sum, model) => sum + model.totalDifference,
                    0
                )
            };
        })
        // Remove tools sem nenhum modelo com diferenças
        .filter(tool => tool.evaluations.length > 0)
        // Ordena tools do maior positivo ao maior negativo
        .sort((a, b) =>
            b.totalDifference - a.totalDifference
        );

    writeFile("just_diff.json", JSON.stringify(filtered, null, 2));
}

main();