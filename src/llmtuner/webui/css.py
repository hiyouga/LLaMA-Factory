CSS = r"""
.modal-box {
  position: fixed !important;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%); /* center horizontally */
  max-width: 1000px;
  max-height: 750px;
  overflow-y: scroll !important;
  background-color: var(--input-background-fill);
  border: 2px solid black !important;
  z-index: 1000;
}

.dark .modal-box {
  border: 2px solid white !important;
}
"""
