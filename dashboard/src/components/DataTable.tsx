import {
  createSortedRowModel,
  rowSortingFeature,
  sortFns,
  tableFeatures,
  useTable,
  type ColumnDef,
  type RowData,
} from "@tanstack/react-table";
import type { ReactNode } from "react";

/** Sortable table on TanStack Table v9 (feature slots). */
export const features = tableFeatures({
  rowSortingFeature,
  sortedRowModel: createSortedRowModel(),
  sortFns,
});
export type Features = typeof features;
export type Col<T extends RowData> = ColumnDef<Features, T, any>; // eslint-disable-line @typescript-eslint/no-explicit-any

export function DataTable<T extends RowData>({ data, columns, caption, onRowClick, selectedKey, rowKey, empty, rowClassName }: {
  data: T[];
  columns: Col<T>[];
  caption: string;
  onRowClick?: (row: T) => void;
  selectedKey?: string | number | null;
  rowKey: (row: T) => string | number;
  empty: ReactNode;
  rowClassName?: (row: T) => string | undefined;
}) {
  const table = useTable({ features, columns, data, getRowId: (r: T) => String(rowKey(r)) });
  const rows = table.getRowModel().rows;
  return (
    <div className="table-wrap">
      <table className="data">
        <caption className="visually-hidden">{caption}</caption>
        <thead>
          {table.getHeaderGroups().map((hg) => (
            <tr key={hg.id}>
              {hg.headers.map((h) => {
                const sorted = h.column.getIsSorted();
                const meta = (h.column.columnDef.meta ?? {}) as { num?: boolean };
                return (
                  <th
                    key={h.id}
                    scope="col"
                    className={meta.num ? "num" : undefined}
                    aria-sort={sorted === "asc" ? "ascending" : sorted === "desc" ? "descending" : undefined}
                  >
                    {h.column.getCanSort() ? (
                      <button type="button" onClick={h.column.getToggleSortingHandler()}>
                        <table.FlexRender header={h} />
                        <span aria-hidden="true">{sorted === "asc" ? " ▴" : sorted === "desc" ? " ▾" : ""}</span>
                      </button>
                    ) : (
                      <table.FlexRender header={h} />
                    )}
                  </th>
                );
              })}
            </tr>
          ))}
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <tr>
              <td colSpan={columns.length}>{empty}</td>
            </tr>
          ) : (
            rows.map((row) => (
              <tr
                key={row.id}
                aria-selected={selectedKey !== undefined ? String(selectedKey) === row.id : undefined}
                onClick={onRowClick ? () => onRowClick(row.original) : undefined}
                className={rowClassName?.(row.original)}
                style={onRowClick ? { cursor: "pointer" } : undefined}
              >
                {row.getAllCells().map((cell) => {
                  const meta = (cell.column.columnDef.meta ?? {}) as { num?: boolean };
                  return (
                    <td key={cell.id} className={meta.num ? "num" : undefined}>
                      <table.FlexRender cell={cell} />
                    </td>
                  );
                })}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}
