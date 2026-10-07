export type ButtonVariant = "primary" | "quiet" | "danger";

export function buttonClass(variant: ButtonVariant = "primary"): string {
  return variant === "primary" ? "btn" : `btn ${variant}`;
}
