@hinode_init
@hinode_event
@hinode_plot
;-----------------------------------------
; Main routine
;-----------------------------------------
pro hinode, reset_state=reset_state
	!p.multi=0
	close,/all
	state = hinode_init(reset_state=reset_state)
	 
	widget_control, state.baseWidget, /REALIZE
	show_images, state
	 
	xmanager, 'hinode', state.baseWidget, EVENT_HANDLER='hinode_Event'

; Free memory
	heap_free, state
	 
end
