/** A preserved screen can retain modal state, but only the visible screen may own a top-layer dialog. */
export function shouldShowPreservedModal(screenVisible: boolean, hasPendingDialog: boolean) {
  return screenVisible && hasPendingDialog
}
