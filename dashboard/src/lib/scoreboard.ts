/**
 * The Test Lab's session scoreboard (D186): every revealed real test transaction counts once.
 * An alert is an E12 decision of step-up or higher (D159).
 *   caught       fraud and alerted          missed       fraud, not alerted
 *   false alarm  legitimate and alerted     correct pass legitimate, not alerted
 */
export interface Scoreboard {
  caught: number;
  missed: number;
  falseAlarms: number;
  correctPasses: number;
  matchesFrozen: number;
  seen: number[];
}

export const emptyScoreboard: Scoreboard = { caught: 0, missed: 0, falseAlarms: 0, correctPasses: 0, matchesFrozen: 0, seen: [] };

export type Outcome = "caught" | "missed" | "false alarm" | "correct pass";

export function outcome(alert: boolean, isFraud: 0 | 1): Outcome {
  if (isFraud === 1) return alert ? "caught" : "missed";
  return alert ? "false alarm" : "correct pass";
}

export function record(board: Scoreboard, id: number, alert: boolean, isFraud: 0 | 1, matchesFrozen: boolean): Scoreboard {
  if (board.seen.includes(id)) return board; // revealing twice does not count twice
  const o = outcome(alert, isFraud);
  return {
    caught: board.caught + (o === "caught" ? 1 : 0),
    missed: board.missed + (o === "missed" ? 1 : 0),
    falseAlarms: board.falseAlarms + (o === "false alarm" ? 1 : 0),
    correctPasses: board.correctPasses + (o === "correct pass" ? 1 : 0),
    matchesFrozen: board.matchesFrozen + (matchesFrozen ? 1 : 0),
    seen: [...board.seen, id],
  };
}
