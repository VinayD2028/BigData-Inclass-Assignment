#===========================================================
# Region_Manager.rb
#-----------------------------------------------------------
# Template for map/story progression checks.
# Keeps progression logic in one place.
#===========================================================

module PokemonWorldChronicles
  module RegionManager
    #-------------------------------------------------------
    # Configuration
    #-------------------------------------------------------
    module Config
      # Example map access requirements.
      # Key: map_id
      # Value: hash of requirements for that map.
      MAP_REQUIREMENTS = {
        # 15 => { badges: 2, event_flag: :bridge_repaired }
      }

      # Enable this during testing if you want to skip lock checks.
      DEBUG_BYPASS = false
    end

    class << self
      # Determine if player can enter a map.
      # Returns [allowed_boolean, message_string]
      def can_enter_map?(map_id)
        return [true, "Debug bypass enabled."] if Config::DEBUG_BYPASS

        req = Config::MAP_REQUIREMENTS[map_id]
        return [true, "No restrictions for this map."] if req.nil?

        # TODO: Replace placeholder checks with Essentials APIs.
        badges_ok = check_badges(req[:badges])
        event_ok  = check_event_flag(req[:event_flag])

        if badges_ok && event_ok
          [true, "Access granted."]
        else
          [false, "You cannot enter yet. Progress the story first."]
        end
      end

      # Placeholder: Check badge requirement.
      def check_badges(required_count)
        return true if required_count.nil?
        # TODO: Hook into Essentials trainer badge count.
        return true
      end

      # Placeholder: Check story/event flag requirement.
      def check_event_flag(flag_symbol)
        return true if flag_symbol.nil?
        # TODO: Hook into Essentials switch/self-switch/global event tracking.
        return true
      end
    end
  end
end
