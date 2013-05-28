;-----------------------------------------
; Event handler
;-----------------------------------------
pro hinode_Event, event
	 widget_control, Event.id, GET_UVALUE=Action
	 widget_control, Event.top, GET_UVALUE=state

	 handler = Event.Top
	 
	 case Action of
	 	'CLICK': begin
			if (event.x ne state.xpos or event.y ne state.ypos) then begin
				
				plot_profiles, state, event.x, event.y
				
				state.xpos = event.x
				state.ypos = event.y
								
			endif
       end
       
       'CONTINUUM': begin
			state.image = 0
			show_images, state
       end
       
       'FLUXDENSITY': begin
			state.image = 1
			show_images, state
       end
       
	endcase

	widget_control, handler, SET_UVALUE=state

end
